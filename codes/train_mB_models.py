#!/usr/bin/env python3
"""
mB: Kavramsal tutarlılık reward'lı GRPO eğitimi.

mA'dan TEK FARK: accuracy_reward yerine ConsistencyReward kullanılır.
  - Orijinal doğru + Benzer doğru → 1.0  (gerçek anlama)
  - Orijinal doğru + Benzer yanlış → 0.5  (şanslı tahmin şüphesi)
  - Orijinal yanlış              → 0.0  (bilmiyor)

Qwen3 gibi thinking destekleyen modellerde --disable-thinking kullanılabilir.

Eğitim: LoRA (r=16, alpha=32, all-linear) + GRPO
"""

import argparse
import json
import os
import re
from pathlib import Path
import math

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


# ============================================================
# Prompt  (mA ile aynı)
# ============================================================
SYSTEM_PROMPT = (
    "Sen Türkçe çoktan seçmeli soruları çözen bir asistansın. "
    "Kısa ve öz muhakemeni yaz, sonra cevabını <answer>X</answer> formatında ver "
    "(X yerine A, B, C veya D harfini koy)."
)

USER_TEMPLATE = """Aşağıdaki çoktan seçmeli soruyu cevaplayın.

Soru: {question}
A) {a}
B) {b}
C) {c}
D) {d}

Kısa ve öz muhakemenizi yazın (gereksiz detaylardan kaçının), sonra cevabınızı şu formatta verin: <answer>X</answer>"""


# ============================================================
# Answer extraction  (mA ile aynı)
# ============================================================
ANSWER_RE = re.compile(r"<answer>\s*([A-Da-d])\s*</answer>", re.IGNORECASE)


def extract_answer(text):
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].upper()


def _completion_text(completion):
    if isinstance(completion, list):
        return completion[0]["content"]
    return completion


# ============================================================
# mB REWARD: ConsistencyReward
# ============================================================
class ConsistencyReward:
    def __init__(
        self,
        num_generations=8,
        similar_reward=0.5,
        disable_thinking=False,
    ):
        self.__name__ = "consistency_reward"
        self.num_generations = num_generations
        self.similar_reward = similar_reward
        self.disable_thinking = disable_thinking
        self.model = None
        self.tokenizer = None
        self.stats = {
            "both_correct": 0,
            "orig_only": 0,
            "sim_only": 0,
            "both_wrong": 0,
            "sim_checked": 0,
        }

    def set_model(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(
        self,
        completions,
        answer_letter,
        similar_prompt_text,
        similar_answer_letter,
        **kwargs,
    ):
        n = len(completions)
        G = self.num_generations
        num_prompts = n // G

        orig_correct = []
        for c, gold in zip(completions, answer_letter):
            text = _completion_text(c)
            pred = extract_answer(text)
            orig_correct.append(pred == gold)

        sim_correct_map = {}
        for p in range(num_prompts):
            start = p * G
            if not any(orig_correct[start:start + G]):
                continue

            sim_ok = self._check_similar(
                similar_prompt_text[start],
                similar_answer_letter[start],
            )
            sim_correct_map[p] = sim_ok
            self.stats["sim_checked"] += 1

        rewards = []
        for i in range(n):
            p = i // G

            if not orig_correct[i]:
                sim_ok = sim_correct_map.get(p)

                if sim_ok is True:
                    self.stats["sim_only"] += 1
                else:
                    self.stats["both_wrong"] += 1

                rewards.append(0.0)
            else:
                sim_ok = sim_correct_map.get(p, False)

                if sim_ok:
                    self.stats["both_correct"] += 1
                    rewards.append(1.0)
                else:
                    self.stats["orig_only"] += 1
                    rewards.append(self.similar_reward)

        return rewards

    def _check_similar(self, sim_prompt_text, sim_gold_letter):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sim_prompt_text},
        ]

        chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }

        if self.disable_thinking:
            chat_template_kwargs["enable_thinking"] = False

        input_text = self.tokenizer.apply_chat_template(
            messages,
            **chat_template_kwargs,
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
        ).to(self.model.device)

        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        if was_training:
            self.model.train()

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        pred = extract_answer(response)
        return pred == sim_gold_letter


# ============================================================
# format_reward  (mA ile aynı)
# ============================================================
def format_reward(completions, **kwargs):
    rewards = []

    for c in completions:
        text = _completion_text(c)
        rewards.append(0.1 if ANSWER_RE.search(text) else 0.0)

    return rewards


# ============================================================
# Data loading
# ============================================================
def load_dataset_from_jsonl(path, limit=None):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            if "similar_question" not in obj:
                continue

            user_msg = USER_TEMPLATE.format(
                question=obj["question"],
                a=obj["choices"][0],
                b=obj["choices"][1],
                c=obj["choices"][2],
                d=obj["choices"][3],
            )

            similar_user_msg = USER_TEMPLATE.format(
                question=obj["similar_question"],
                a=obj["similar_choices"][0],
                b=obj["similar_choices"][1],
                c=obj["similar_choices"][2],
                d=obj["similar_choices"][3],
            )

            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "answer_letter": obj["answer_letter"],
                "similar_prompt_text": similar_user_msg,
                "similar_answer_letter": obj["similar_answer_letter"],
            })

            if limit is not None and len(rows) >= limit:
                break

    return Dataset.from_list(rows)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default="models/Qwen2.5-3B-Instruct",
    )
    ap.add_argument(
        "--data",
        default="data/mmlu-tr-selected-extended/train_synthetic.jsonl",
    )
    ap.add_argument(
        "--output-dir",
        default="results/mB_full",
    )
    ap.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Qwen3 gibi thinking destekleyen modellerde thinking modunu kapatır.",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--similar-reward", type=float, default=0.5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--gen-batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-completion-len", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[VERİ] {args.data}")

    if args.limit:
        print(f"  Limit: {args.limit}")

    dataset = load_dataset_from_jsonl(
        args.data,
        limit=args.limit,
    )

    print(f"  Boyut: {len(dataset)}\n")

    prompts_per_forward = (
        args.per_device_batch // args.num_generations
    )
    effective_prompt_batch = (
        prompts_per_forward * args.grad_accum
    )

    total_steps = max(
        1,
        math.ceil(
            (len(dataset) / effective_prompt_batch)
            * args.epochs
        ),
    )

    save_steps = max(1, total_steps // 2)

    print(
        f"[ADIM] per_device_batch={args.per_device_batch} = "
        f"{prompts_per_forward} prompt × "
        f"{args.num_generations} rollout"
    )
    print(
        f"       grad_accum={args.grad_accum}, "
        f"effective batch={effective_prompt_batch}"
    )
    print(
        f"       toplam step={total_steps}, "
        f"checkpoint @ {save_steps}\n"
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    consistency_reward = ConsistencyReward(
        num_generations=args.num_generations,
        similar_reward=args.similar_reward,
        disable_thinking=args.disable_thinking,
    )

    print(
        f"[mB] ✓✓→1.0 | "
        f"✓✗→{args.similar_reward} | "
        f"✗→0.0\n"
    )

    grpo_kwargs = {}

    if args.disable_thinking:
        grpo_kwargs["chat_template_kwargs"] = {
            "enable_thinking": False,
        }

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        loss_type="grpo",
        num_generations=args.num_generations,
        generation_batch_size=(
            args.gen_batch_size
            or args.per_device_batch
        ),
        max_completion_length=args.max_completion_len,
        temperature=args.temperature,
        beta=args.beta,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        report_to="none",
        seed=args.seed,
        **grpo_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("[TRAINER] hazırlanıyor...")

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[
            consistency_reward,
            format_reward,
        ],
        peft_config=peft_config,
    )

    consistency_reward.set_model(
        trainer.model,
        tokenizer,
    )

    print("[mB] model & tokenizer bağlandı.\n")

    print("[EĞİTİM] başlıyor...\n")
    trainer.train()

    s = consistency_reward.stats

    total = (
        s["both_correct"]
        + s["orig_only"]
        + s["sim_only"]
        + s["both_wrong"]
    )

    print(f"\n{'=' * 50}")
    print(f"[mB İSTATİSTİK] Toplam {total} rollout:")
    print(f"  ✓✓ {s['both_correct']:5d} → 1.0")
    print(
        f"  ✓✗ {s['orig_only']:5d} "
        f"→ {args.similar_reward}"
    )
    print(f"  ✗✓ {s['sim_only']:5d} → 0.0")
    print(f"  ✗✗ {s['both_wrong']:5d} → 0.0")
    print(
        f"  Benzer soru soruldu: "
        f"{s['sim_checked']} kez"
    )
    print(f"{'=' * 50}")

    final_dir = os.path.join(
        args.output_dir,
        "final",
    )

    print(f"\n[KAYIT] {final_dir}")
    trainer.save_model(final_dir)

    print("\n[BİTTİ]")


if __name__ == "__main__":
    main()