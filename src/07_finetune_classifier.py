"""
Step 7: Fine-tuned Classifier
Full fine-tune of DistilBERT on AG News 4-class classification.

IMPORTANT — train/eval split discipline:
  - TRAINING:   AG News `train` split (120,000 items) from HuggingFace.
  - EVALUATION: the exact 1,000 items in data/ag_news_sample.csv (drawn from the
    AG News `test` split in 01_data_prep.py). This is the SAME evaluation set
    used by human-consensus and Claude LLM annotations in steps 02-05, so all
    three methods are compared on identical inputs. We do NOT fine-tune on
    these 1,000 items — doing so would leak the eval set.
"""

import json
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from utils import CATEGORIES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "distilbert_agnews"

MODEL_NAME = "distilbert-base-uncased"
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}
MAX_LENGTH = 128
NUM_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 5e-5


def tokenize_fn(tokenizer):
    def _fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)
    return _fn


def train_model():
    print("Loading AG News TRAIN split (120K items) for fine-tuning...")
    # NOTE: we explicitly load the train split. The 1,000-item eval set lives in
    # data/ag_news_sample.csv (sampled from the AG News test split) — never used here.
    train_ds = load_dataset("ag_news", split="train")
    print(f"Training set size: {len(train_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_tok = train_ds.map(tokenize_fn(tokenizer), batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=LABEL_MAP,
        label2id=LABEL_TO_ID,
    )

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_steps=200,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("Fine-tuning DistilBERT (full fine-tune, not LoRA)...")
    trainer.train()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"Model saved to {MODEL_DIR}")
    return model, tokenizer


def load_or_train():
    if (MODEL_DIR / "config.json").exists():
        print(f"Loading fine-tuned model from {MODEL_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        return model, tokenizer
    return train_model()


def evaluate_on_eval_set(model, tokenizer):
    """Evaluate on the 1,000-item eval set used by all three methods."""
    eval_path = DATA_DIR / "ag_news_sample.csv"
    df = pd.read_csv(eval_path)
    print(f"\nEvaluating on the shared 1,000-item eval set ({eval_path})")
    print("(Same items used for human-consensus and Claude LLM evaluation.)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preds = []
    latencies = []
    with torch.no_grad():
        for text in df["text"].tolist():
            enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            t0 = time.perf_counter()
            logits = model(**enc).logits
            pred_id = int(torch.argmax(logits, dim=-1).item())
            latencies.append(time.perf_counter() - t0)
            preds.append(LABEL_MAP[pred_id])

    df["finetuned_model"] = preds
    y_true = df["true_label"].tolist()

    accuracy = accuracy_score(y_true, preds)
    f1_per_class = f1_score(y_true, preds, labels=CATEGORIES, average=None)
    cm = confusion_matrix(y_true, preds, labels=CATEGORIES)
    latency_ms = float(np.mean(latencies) * 1000)

    print("\n" + "=" * 60)
    print("FINE-TUNED DISTILBERT — EVALUATION (1,000 items)")
    print("=" * 60)
    print(f"  Accuracy:            {accuracy:.4f}")
    print(f"  Mean latency/sample: {latency_ms:.2f} ms  (device={device})")
    print("\n  Per-class F1:")
    for cat, f in zip(CATEGORIES, f1_per_class):
        print(f"    {cat:10s}: {f:.4f}")
    print("\n" + classification_report(y_true, preds, labels=CATEGORIES))

    # Persist predictions into annotations_all.csv (joined on item_id) so
    # steps 05/06 can consume them alongside human + LLM annotations.
    all_path = DATA_DIR / "annotations_all.csv"
    if all_path.exists():
        ann = pd.read_csv(all_path)
        if "finetuned_model" in ann.columns:
            ann = ann.drop(columns=["finetuned_model"])
        ann = ann.merge(df[["item_id", "finetuned_model"]], on="item_id", how="left")
        ann.to_csv(all_path, index=False)
        print(f"\nMerged finetuned_model column into {all_path}")
    else:
        print(f"\nWARNING: {all_path} not found — run steps 1-3 first to combine annotations.")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model": MODEL_NAME,
        "training_set": "ag_news train split (120,000 items)",
        "eval_set": "data/ag_news_sample.csv (1,000 items — same as human/LLM eval)",
        "accuracy": float(accuracy),
        "f1_per_class": {cat: float(f) for cat, f in zip(CATEGORIES, f1_per_class)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": CATEGORIES,
        "latency_ms_per_sample": latency_ms,
        "device": device,
    }
    out_path = OUTPUTS_DIR / "finetuned_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_path}")


def main():
    model, tokenizer = load_or_train()
    evaluate_on_eval_set(model, tokenizer)


if __name__ == "__main__":
    main()
