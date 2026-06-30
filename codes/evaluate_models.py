#!/usr/bin/env python3
"""
Base model ile mA veya mB LoRA adapter'ını aynı Türkçe MCQ test setinde
karşılaştırır.

Qwen3 gibi thinking destekleyen modellerde --disable-thinking kullanılabilir.

Çıktılar:

  output_dir/
    base_predictions.jsonl
    <run_name>_final_predictions.jsonl
    comparison_summary.json
"""

import argparse
import gc
import json
import math
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "Sen Türkçe çoktan seçmeli soruları çözen bir asistansın. "
    "Önce kısa muhakemeni yaz, sonra cevabını <answer>X</answer> formatında ver "
    "(X yerine A, B, C veya D harfini koy)."
)

USER_TEMPLATE = """Aşağıdaki çoktan seçmeli soruyu cevaplayın.

Soru: {question}
A) {a}
B) {b}
C) {c}
D) {d}

Önce muhakemenizi yazın, sonra cevabınızı şu formatta verin: <answer>X</answer>"""

ANSWER_RE = re.compile(
    r"<answer>\s*([A-Da-d])\s*</answer>",
    re.IGNORECASE,
)


def extract_answer(text: str):
    matches = ANSWER_RE.findall(text)
    return matches[-1].upper() if matches else None


def load_jsonl(path: str, limit=None):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)
            rows.append(obj)

            if limit is not None and len(rows) >= limit:
                break

    return rows


def build_messages(row):
    choices = row["choices"]

    user_message = USER_TEMPLATE.format(
        question=row["question"],
        a=choices[0],
        b=choices[1],
        c=choices[2],
        d=choices[3],
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


@torch.inference_mode()
def evaluate(
    model,
    tokenizer,
    rows,
    model_name,
    output_path,
    batch_size=8,
    max_new_tokens=512,
    disable_thinking=False,
):
    model.eval()

    results = []
    n_correct = 0
    n_formatted = 0

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        conversations = [build_messages(row) for row in batch_rows]

        chat_template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
            "padding": True,
        }

        if disable_thinking:
            chat_template_kwargs["enable_thinking"] = False

        encoded = tokenizer.apply_chat_template(
            conversations,
            **chat_template_kwargs,
        )

        encoded = {
            key: value.to(model.device)
            for key, value in encoded.items()
        }

        prompt_width = encoded["input_ids"].shape[1]

        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_ids = outputs[:, prompt_width:]

        texts = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        for row, text in zip(batch_rows, texts):
            prediction = extract_answer(text)
            gold = row["answer_letter"].upper()
            correct = prediction == gold

            n_correct += int(correct)
            n_formatted += int(prediction is not None)

            results.append({
                "id": row.get("id"),
                "subject": row.get("subject"),
                "gold": gold,
                "prediction": prediction,
                "correct": correct,
                "formatted": prediction is not None,
                "completion": text,
            })

        processed = min(start + batch_size, len(rows))
        running_accuracy = n_correct / processed

        print(
            f"[{model_name}] "
            f"{processed}/{len(rows)} | "
            f"accuracy={running_accuracy:.4f}"
        )

    accuracy = n_correct / len(rows)
    format_rate = n_formatted / len(rows)

    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(
        f"\n[{model_name}] "
        f"accuracy={accuracy:.4f} ({n_correct}/{len(rows)}) | "
        f"format_rate={format_rate:.4f}\n"
    )

    return {
        "name": model_name,
        "accuracy": accuracy,
        "correct": n_correct,
        "total": len(rows),
        "format_rate": format_rate,
        "results": results,
    }


def exact_mcnemar_p_value(base_results, final_results):
    """
    Paired exact McNemar testi:
      improved: base yanlış, final doğru
      degraded: base doğru, final yanlış
    """
    improved = 0
    degraded = 0

    for base, final in zip(base_results, final_results):
        if not base["correct"] and final["correct"]:
            improved += 1
        elif base["correct"] and not final["correct"]:
            degraded += 1

    discordant = improved + degraded

    if discordant == 0:
        return improved, degraded, 1.0

    k = min(improved, degraded)

    lower_tail = sum(
        math.comb(discordant, i)
        for i in range(k + 1)
    ) / (2 ** discordant)

    p_value = min(1.0, 2.0 * lower_tail)

    return improved, degraded, p_value


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-model",
        default="models/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--adapter",
        default="results/mA_full/final",
    )
    parser.add_argument(
        "--test-data",
        default="data/mmlu-tr-selected-extended/test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="results/evaluation",
    )
    parser.add_argument(
        "--run-name",
        choices=["mA", "mB"],
        required=True,
        help="Değerlendirilen adapter koşulu: mA veya mB.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Qwen3 gibi thinking destekleyen modellerde thinking modunu kapatır.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.test_data, limit=args.limit)
    print(f"[TEST] {len(rows)} örnek yüklendi.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[BASE] Model yükleniyor...")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    base_model.config.pad_token_id = tokenizer.pad_token_id

    base_metrics = evaluate(
        model=base_model,
        tokenizer=tokenizer,
        rows=rows,
        model_name="base",
        output_path=output_dir / "base_predictions.jsonl",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        disable_thinking=args.disable_thinking,
    )

    print("[ADAPTER] LoRA adapter yükleniyor...")

    final_model = PeftModel.from_pretrained(
        base_model,
        args.adapter,
        is_trainable=False,
    )

    final_model.eval()

    final_name = f"{args.run_name}_final"

    final_metrics = evaluate(
        model=final_model,
        tokenizer=tokenizer,
        rows=rows,
        model_name=final_name,
        output_path=output_dir / f"{final_name}_predictions.jsonl",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        disable_thinking=args.disable_thinking,
    )

    improved, degraded, p_value = exact_mcnemar_p_value(
        base_metrics["results"],
        final_metrics["results"],
    )

    unchanged_correct = sum(
        b["correct"] and f["correct"]
        for b, f in zip(
            base_metrics["results"],
            final_metrics["results"],
        )
    )

    unchanged_wrong = sum(
        (not b["correct"]) and (not f["correct"])
        for b, f in zip(
            base_metrics["results"],
            final_metrics["results"],
        )
    )

    summary = {
        "run_name": args.run_name,
        "n_test": len(rows),
        "base": {
            key: value
            for key, value in base_metrics.items()
            if key != "results"
        },
        final_name: {
            key: value
            for key, value in final_metrics.items()
            if key != "results"
        },
        "accuracy_difference": (
            final_metrics["accuracy"] - base_metrics["accuracy"]
        ),
        "base_wrong_final_correct": improved,
        "base_correct_final_wrong": degraded,
        "both_correct": unchanged_correct,
        "both_wrong": unchanged_wrong,
        "exact_mcnemar_p_value": p_value,
    }

    summary_path = output_dir / "comparison_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Base accuracy      : {base_metrics['accuracy']:.4f}")
    print(f"{final_name} accuracy : {final_metrics['accuracy']:.4f}")
    print(
        "Fark               : "
        f"{summary['accuracy_difference']:+.4f}"
    )
    print(f"Yanlış → doğru     : {improved}")
    print(f"Doğru → yanlış     : {degraded}")
    print(f"McNemar p-value    : {p_value:.6f}")
    print(f"Özet dosyası       : {summary_path}")
    print("=" * 60)

    del final_model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()