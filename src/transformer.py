"""Transformer approach: fine-tune XLM-RoBERTa (multilingual, covers Arabic and English).

Same train / validation data as the baseline, so the comparison is fair.
The checkpoint with the best validation macro-F1 is kept in models/xlmr-sentiment/.

Also scores an off-the-shelf pretrained sentiment model (no training on our data)
as a third reference point: models/zero_shot_val.csv.

Needs a GPU for reasonable speed (~20 min on an RTX 3060 laptop GPU).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, pipeline)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "models" / "xlmr-sentiment"
BASE_MODEL = "xlm-roberta-base"
ZERO_SHOT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
LABELS = ["negative", "mixed", "positive"]
MAX_LEN = 192
SEED = 42


def to_dataset(df, tok):
    ds = Dataset.from_pandas(pd.DataFrame({"text": df["text"], "label": df["label"].map(LABELS.index)}))
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=["text"])


def metrics(eval_pred):
    logits, y = eval_pred
    return {"macro_f1": f1_score(y, logits.argmax(-1), average="macro")}


def fine_tune():
    train, val = pd.read_parquet(DATA / "train.parquet"), pd.read_parquet(DATA / "val.parquet")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    args = TrainingArguments(
        output_dir=os.environ.get("SENTIMENT_CHECKPOINT_DIR", str(ROOT / "models" / "_checkpoints")), num_train_epochs=3, learning_rate=2e-5,
        per_device_train_batch_size=16, per_device_eval_batch_size=64, warmup_steps=340, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="macro_f1", fp16=torch.cuda.is_available(), logging_steps=100,
        report_to=[], seed=SEED, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=to_dataset(train, tok), eval_dataset=to_dataset(val, tok),
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    t = time.time()
    trainer.train()
    best = trainer.evaluate()
    trainer.save_model(str(OUT))
    tok.save_pretrained(str(OUT))
    log = [h for h in trainer.state.log_history if "eval_macro_f1" in h]
    (ROOT / "models" / "transformer_info.json").write_text(json.dumps(
        {"base_model": BASE_MODEL, "max_len": MAX_LEN, "epochs": 3, "lr": 2e-5, "batch_size": 16,
         "train_minutes": round((time.time() - t) / 60, 1), "best_val_macro_f1": best["eval_macro_f1"],
         "val_by_epoch": [{"epoch": h["epoch"], "macro_f1": h["eval_macro_f1"], "loss": h["eval_loss"]} for h in log]},
        indent=2))
    print("best val macro-F1:", best["eval_macro_f1"])


def zero_shot():
    """Off-the-shelf multilingual sentiment model (trained on tweets), no fine-tuning."""
    val = pd.read_parquet(DATA / "val.parquet")
    clf = pipeline("text-classification", model=ZERO_SHOT_MODEL, device=0 if torch.cuda.is_available() else -1,
                   truncation=True, max_length=MAX_LEN)
    out = clf(val["text"].tolist(), batch_size=64)
    val["pred"] = [o["label"].lower().replace("neutral", "mixed") for o in out]
    val[["lang", "label", "pred"]].to_csv(ROOT / "models" / "zero_shot_val.csv", index=False)
    print("zero-shot val macro-F1:", f1_score(val["label"], val["pred"], average="macro"))


if __name__ == "__main__":
    zero_shot()
    fine_tune()
