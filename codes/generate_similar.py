#!/usr/bin/env python3
"""
mB pipeline için: train setindeki her MCQ sorusu için aynı kavramı test eden
sentetik bir benzer soru üretir.

Özellikler:
- Resume: yarıda kalırsa kaldığı yerden devam eder (output'taki id'leri okur)
- Her satır anında flush (kopma olursa veri kaybı yok)
- Retry: API hataları için exponential backoff, parse hataları için regen
- tqdm ile progress
- --limit ile küçük scale test (örn. --limit 3)
"""

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


# -------- Prompt --------
SYSTEM_PROMPT = (
    "Sen Türkçe çoktan seçmeli soru (MCQ) hazırlayan kıdemli bir eğitim materyali "
    "uzmanısın. Sana verilen bir orijinal soruyu inceleyip, AYNI kavramı/beceriyi "
    "test eden ama farklı sayı, senaryo veya parametrelerle yepyeni bir MCQ "
    "üretirsin. Yüzeysel kelime değişikliği değil, gerçek bir varyant üretirsin. "
    "Çıktın her zaman geçerli ve parse edilebilir saf JSON olur; ek açıklama, "
    "markdown veya kod bloğu kullanmazsın."
)

USER_TEMPLATE = """Aşağıdaki çoktan seçmeli soruyu incele ve AYNI kavramı test eden YENİ bir soru üret.

KURALLAR:
1. Soru aynı konuyu/kavramı/beceriyi test etmeli, ama farklı sayılar/senaryolar/değişkenler kullanmalı.
2. Yüzeysel kelime değişikliği YAPMA — gerçek bir kavramsal varyant üret.
3. Tam olarak 4 şık olsun, yalnızca biri doğru olsun.
4. Yanlış şıklar makul ve birbirinden ayırt edilebilir olmalı (gerçekçi tuzaklar, rastgele saçma değerler değil).
5. Türkçe ve dilbilgisi olarak doğru olmalı.
6. Zorluk seviyesi orijinaline benzer olmalı.
7. Doğru cevabın orijinaldekiyle aynı harfte (A/B/C/D) olması ZORUNLU değil; doğal olan harf hangisiyse o olsun.
8. ŞIK FORMATI: "choices" listesindeki her eleman SADECE şıkkın metnini içermeli. Başına "A)", "B)", "A.", "1)" gibi etiket, harf, numara veya işaret EKLEME. Sıra zaten listedeki konumdan belli. Örn: DOĞRU → "Rezonans"   YANLIŞ → "B) Rezonans"
9. HESAPLAMA DOĞRULUĞU: "rationale" alanı SADECE 1-2 cümlelik kısa ve kesin bir gerekçe olmalı. Ara hesap, deneme-yanılma, düzeltme veya alternatif çözüm YAZMA. "answer_letter" ile işaretlediğin şık, rationale'deki sonuçla birebir uyuşmalı.
KONU: {subject}

ORİJİNAL SORU:
{question}

ŞIKLAR:
A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

DOĞRU CEVAP: {answer_letter}

Çıktıyı SADECE aşağıdaki JSON şemasında ver, başka hiçbir şey yazma (markdown ya da ```json bloğu YOK). Dikkat: "choices" elemanları SADECE metin içerir, harf/etiket içermez:

{{
  "question": "yeni soru metni",
  "choices": ["birinci şıkkın yalnızca metni", "ikinci şıkkın yalnızca metni", "üçüncü şıkkın yalnızca metni", "dördüncü şıkkın yalnızca metni"],
  "answer_letter": "X",
  "rationale": "doğru cevabın 1-2 cümlelik net ve tutarlı gerekçesi"
}}
"""


# -------- I/O helpers --------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_ids(path):
    """Output dosyası varsa içindeki id'leri seteye koy (resume için)."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(obj["id"])
            except json.JSONDecodeError:
                # Yarım kalmış son satır olabilir; sessizce atla
                continue
    return done


# -------- JSON parsing --------
def extract_json(text):
    """Modelin çıktısından JSON nesnesini çek. <think>...</think>, ```json ```
    veya prosa açıklama olsa da en dıştaki { } arasını dener."""
    raw = text
    # Qwen3 thinking bloklarını temizle (kapalı VEYA açık kalmış olabilir)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Açık kalmış (max_tokens'a yetişmemiş) thinking bloğunu da at
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    # ```json ... ``` ya da ``` ... ``` temizliği
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # En dıştaki { } yakala
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        snippet = raw[:400].replace("\n", "\\n")
        raise ValueError(f"JSON bulunamadı. Ham yanıt (ilk 400ch): {snippet!r}")
    return json.loads(text[start : end + 1])


_LABEL_PREFIX_RE = re.compile(r"^\s*[\(\[]?[A-D1-4][\)\.\:\-](?:\s+|$)")


def _strip_label_prefix(text):
    """Şık metninin başına eklenmiş 'A)', 'A.', '(A)', '1)' gibi etiketleri kırp.
    Model prompt'taki kurala uymadığında defansif temizlik."""
    if not isinstance(text, str):
        return text
    cleaned = _LABEL_PREFIX_RE.sub("", text, count=1).strip()
    # Tamamen boşalmasın diye fallback
    return cleaned if cleaned else text.strip()


def validate_similar(obj):
    """Üretilen sentetik soru sağlam mı? Değilse ValueError fırlat.
    Yan etki: choices'taki olası harf prefix'leri (A), B) ...) temizlenir."""
    if not isinstance(obj, dict):
        raise ValueError("dict bekleniyor")
    for k in ("question", "choices", "answer_letter"):
        if k not in obj:
            raise ValueError(f"eksik alan: {k}")
    if not isinstance(obj["question"], str) or len(obj["question"].strip()) < 5:
        raise ValueError("question kısa veya str değil")
    if not isinstance(obj["choices"], list) or len(obj["choices"]) != 4:
        raise ValueError("choices 4 elemanlı liste olmalı")
    # Harf prefix'lerini temizle (model kurala uymadıysa son savunma hattı)
    obj["choices"] = [_strip_label_prefix(c) for c in obj["choices"]]
    for i, c in enumerate(obj["choices"]):
        if not isinstance(c, str) or not c.strip():
            raise ValueError(f"choice[{i}] boş veya str değil")
    if obj["answer_letter"] not in ("A", "B", "C", "D"):
        raise ValueError(f"answer_letter geçersiz: {obj['answer_letter']!r}")
    return obj


# -------- API call --------
def call_model(client, model, system, user, *, temperature, top_p, top_k,
               presence_penalty, max_tokens, enable_thinking):
    """Tek çağrı; üst seviyede retry sarmalanacak.

    Qwen3.5 thinking'i kapatma resmi yolu: extra_body içinde
    chat_template_kwargs.enable_thinking=False. /no_think direktifi DESTEKLENMİYOR.
    """
    extra_body = {
        "top_k": top_k,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return resp.choices[0].message.content


def generate_with_retry(client, model, user_prompt, *, api_retries=5,
                        parse_retries=3, enable_thinking=False, max_tokens=4096):
    """API hatasına exponential backoff, parse hatasına regen.

    Sampling parametreleri Qwen3.5 dokümanının önerisine göre seçilir:
      - thinking ON  : T=0.6, top_p=0.95, top_k=20, presence_penalty=0.0
      - thinking OFF : T=0.7, top_p=0.8,  top_k=20, presence_penalty=1.5
    """
    if enable_thinking:
        base_t, top_p, top_k, pp = 0.6, 0.95, 20, 0.0
    else:
        base_t, top_p, top_k, pp = 0.7, 0.8, 20, 1.5

    last_err = None
    for attempt in range(parse_retries):
        text = None
        for api_attempt in range(api_retries):
            try:
                # Her parse retry'da sıcaklığı hafif yükselt (çeşitlilik için)
                t = min(base_t + 0.1 * attempt, 1.0)
                text = call_model(
                    client, model, SYSTEM_PROMPT, user_prompt,
                    temperature=t, top_p=top_p, top_k=top_k,
                    presence_penalty=pp, max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                )
                break
            except Exception as e:
                last_err = e
                sleep_s = min(2 ** api_attempt, 30)
                tqdm.write(f"[api retry {api_attempt+1}/{api_retries}] {type(e).__name__}: {e} — {sleep_s}s bekle")
                time.sleep(sleep_s)
        if text is None:
            raise RuntimeError(f"API tamamen başarısız: {last_err}")

        try:
            obj = extract_json(text)
            return validate_similar(obj)
        except Exception as e:
            last_err = e
            tqdm.write(f"[parse retry {attempt+1}/{parse_retries}] {type(e).__name__}: {e}")
            continue
    raise ValueError(f"Parse/validate başarısız: {last_err}")


# -------- Main pipeline --------
def build_user_prompt(row):
    return USER_TEMPLATE.format(
        subject=row.get("subject", "unknown"),
        question=row["question"],
        choice_a=row["choices"][0],
        choice_b=row["choices"][1],
        choice_c=row["choices"][2],
        choice_d=row["choices"][3],
        answer_letter=row["answer_letter"],
    )


def make_output_record(orig, similar):
    """Orijinali koru + similar_* alanları ekle."""
    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    return {
        "id": orig["id"],
        "subject": orig.get("subject"),
        "orig_split": orig.get("orig_split"),
        "question": orig["question"],
        "choices": orig["choices"],
        "answer": orig["answer"],
        "answer_letter": orig["answer_letter"],
        "similar_question": similar["question"].strip(),
        "similar_choices": [c.strip() for c in similar["choices"]],
        "similar_answer_letter": similar["answer_letter"],
        "similar_answer": letter_to_idx[similar["answer_letter"]],
        "similar_rationale": similar.get("rationale", "").strip(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/arf/scratch/proj29/aekiz/compt_semantics/p1/data/mmlu-tr-selected_extended/train.jsonl")
    ap.add_argument("--output", default="/arf/scratch/proj29/aekiz/compt_semantics/p1/data/mmlu-tr-selected_extended/train_synthetic.jsonl")
    ap.add_argument("--failed", default=None, help="Başarısız sorular log dosyası (default: <output>.failed.jsonl)")
    ap.add_argument("--model", default="qwen3-397b-terminus")
    ap.add_argument("--base-url", default="http://kolyoz38:30000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--limit", type=int, default=None, help="Sadece ilk N soruyu işle (test için, örn. --limit 3)")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="Üretim için max token. Thinking açıksa 8192+ önerilir.")
    ap.add_argument("--thinking", action="store_true",
                    help="Qwen3.5 thinking modunu aç. Default KAPALI çünkü MCQ üretimi "
                         "için reasoning gereksiz ve 5-10x daha yavaş olur. "
                         "Thinking için chat_template_kwargs.enable_thinking=True gönderilir.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    failed_path = Path(args.failed) if args.failed else out_path.with_suffix(".failed.jsonl")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: hangileri zaten yazılmış?
    done = load_done_ids(out_path)
    failed_done = load_done_ids(failed_path)
    skip_ids = done | failed_done  # failed olanları da tekrar denemiyoruz (istersen failed dosyayı silersin)

    # Tüm satırları oku
    all_rows = list(read_jsonl(in_path))
    if args.limit is not None:
        all_rows = all_rows[: args.limit]

    # İşlenecekleri filtrele
    todo = [r for r in all_rows if r["id"] not in skip_ids]

    print(f"Input          : {in_path}")
    print(f"Output         : {out_path}")
    print(f"Failed log     : {failed_path}")
    print(f"Model          : {args.model} @ {args.base_url}")
    print(f"Thinking modu  : {'AÇIK' if args.thinking else 'KAPALI (önerilen)'}")
    print(f"Toplam soru    : {len(all_rows)}")
    print(f"Zaten yazılmış : {len(done)}")
    print(f"Önceden hata   : {len(failed_done)} (yeniden denemek istersen failed dosyayı sil)")
    print(f"İşlenecek      : {len(todo)}")
    if args.limit is not None:
        print(f"--limit aktif  : ilk {args.limit} soru")
    print()

    if not todo:
        print("Yapılacak bir şey yok, çıkılıyor.")
        return

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # Ctrl+C için temiz çıkış flag'i
    stop_flag = {"stop": False}
    def handle_sigint(signum, frame):
        if stop_flag["stop"]:
            print("\n[İkinci Ctrl+C, hemen çıkılıyor]")
            sys.exit(130)
        stop_flag["stop"] = True
        print("\n[Ctrl+C alındı, mevcut soru bitince çıkacak. Bir daha basarsan hemen çıkar.]")
    signal.signal(signal.SIGINT, handle_sigint)

    n_ok, n_fail = 0, 0
    # `a` modu + her yazımdan sonra flush + fsync ile dayanıklılık
    with open(out_path, "a", encoding="utf-8") as fout, \
         open(failed_path, "a", encoding="utf-8") as ffail:
        pbar = tqdm(todo, desc="Üretiliyor", unit="q")
        for row in pbar:
            if stop_flag["stop"]:
                break
            try:
                user_prompt = build_user_prompt(row)
                similar = generate_with_retry(
                    client, args.model, user_prompt,
                    max_tokens=args.max_tokens,
                    enable_thinking=args.thinking,
                )
                rec = make_output_record(row, similar)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                n_ok += 1
            except Exception as e:
                err_rec = {
                    "id": row["id"],
                    "subject": row.get("subject"),
                    "error": f"{type(e).__name__}: {e}",
                }
                ffail.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                ffail.flush()
                os.fsync(ffail.fileno())
                n_fail += 1
                tqdm.write(f"[FAIL] {row['id']}: {e}")
            pbar.set_postfix(ok=n_ok, fail=n_fail)

    print()
    print(f"Bitti. Başarılı: {n_ok}, Başarısız: {n_fail}")
    print(f"Çıktı: {out_path}")
    if n_fail:
        print(f"Hatalar: {failed_path}")


if __name__ == "__main__":
    main()