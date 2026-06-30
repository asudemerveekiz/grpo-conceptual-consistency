#!/usr/bin/env python3
"""
mA: Standart GRPO ile Türkçe MCQ eğitimi (baseline).

Reward fonksiyonları (TRL bunları otomatik toplar):
  - accuracy_reward: doğruysa 1.0, yanlışsa 0.0
  - format_reward  : <answer>X</answer> formatına uyuyorsa +0.1

Eğitim: LoRA (r=16, alpha=32, all-linear) + GRPO
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


# ============================================================
# Prompt
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
# Answer extraction
# ============================================================
ANSWER_RE = re.compile(r"<answer>\s*([A-Da-d])\s*</answer>", re.IGNORECASE)


def extract_answer(text):
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].upper()


# ============================================================
# Reward functions
# ============================================================
def _completion_text(completion):
    """Conversational prompt → completion = [{'role':'assistant','content':'...'}]
    Plain prompt → completion = 'str'. İkisini de handle eder."""
    if isinstance(completion, list):
        return completion[0]["content"]
    return completion


def accuracy_reward(completions, answer_letter, **kwargs):
    """Doğruysa 1.0, yanlışsa 0.0. Format yoksa otomatik 0.0 (pred=None)."""
    rewards = []
    for c, gold in zip(completions, answer_letter):
        text = _completion_text(c)
        pred = extract_answer(text)
        rewards.append(1.0 if pred == gold else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    """<answer>X</answer> formatına uyuyorsa 0.1, yoksa 0.0."""
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
            user_msg = USER_TEMPLATE.format(
                question=obj["question"],
                a=obj["choices"][0],
                b=obj["choices"][1],
                c=obj["choices"][2],
                d=obj["choices"][3],
            )

            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "answer_letter": obj["answer_letter"],
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
        default="results/mA_full",
    )
    ap.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Qwen3 gibi thinking destekleyen modellerde thinking modunu kapatır.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Train set boyutu (deneme için, default: tümü)",
    )
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="GRPO rollout sayısı (G)",
    )
    ap.add_argument(
        "--per-device-batch",
        type=int,
        default=32,
        help=(
            "Forward başına completion sayısı. num_generations'ın katı olmalı. "
            "32 = 4 prompt × 8 rollout paralel."
        ),
    )
    ap.add_argument(
        "--grad-accum",
        type=int,
        default=2,
        help=(
            "Effective prompt batch = "
            "(per_device_batch / num_generations) × grad_accum. "
            "Default 32/8 × 2 = 8 prompt per optim step."
        ),
    )
    ap.add_argument(
        "--gen-batch-size",
        type=int,
        default=None,
        help="Generation sırasında paralel batch (default = per_device_batch).",
    )
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-completion-len", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="GRPO KL coefficient",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # --- Veri ---
    print(f"[VERİ] {args.data}")
    if args.limit:
        print(f"  Limit: {args.limit} sample")
    dataset = load_dataset_from_jsonl(args.data, limit=args.limit)
    print(f"  Boyut: {len(dataset)}\n")

    # --- Step hesabı ---
    prompts_per_forward = args.per_device_batch // args.num_generations
    effective_prompt_batch = prompts_per_forward * args.grad_accum
    total_steps = max(
        1,
        math.ceil((len(dataset) / effective_prompt_batch) * args.epochs),
    )
    save_steps = max(1, total_steps // 2)

    print(
        f"[ADIM] per_device_batch = {args.per_device_batch} completion = "
        f"{prompts_per_forward} prompt × {args.num_generations} rollout"
    )
    print(f"       grad_accum = {args.grad_accum}")
    print(
        f"       effective prompt batch = "
        f"{prompts_per_forward} × {args.grad_accum} = {effective_prompt_batch}"
    )
    print(f"       toplam optim step = {total_steps}")
    print(f"       checkpoint @ step {save_steps} (~%50) ve bitişte\n")

    # --- LoRA ---
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    # --- GRPO config ---
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        # Optimization
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        # GRPO
        loss_type="grpo",
        num_generations=args.num_generations,
        generation_batch_size=args.gen_batch_size or args.per_device_batch,
        max_completion_length=args.max_completion_len,
        temperature=args.temperature,
        beta=args.beta,
        chat_template_kwargs={
            "enable_thinking": not args.disable_thinking,
        },
        # Precision / memory
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Logging / saving
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        report_to="none",
        # Repro
        seed=args.seed,
    )

    # --- Trainer ---
    print("[TRAINER] hazırlanıyor (model yükleniyor)...")
    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[accuracy_reward, format_reward],
        peft_config=peft_config,
    )

    # --- Eğit ---
    print("[EĞİTİM] başlıyor...\n")
    trainer.train()

    # --- Final adapter ---
    final_dir = os.path.join(args.output_dir, "final")
    print(f"\n[KAYIT] final adapter → {final_dir}")
    trainer.save_model(final_dir)

    print("\n[BİTTİ]")


if __name__ == "__main__":
    main()