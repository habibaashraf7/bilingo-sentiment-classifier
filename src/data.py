"""Build balanced Arabic + English train / validation / test sets with one label scheme.

Labels: negative, mixed, positive.
  * Arabic 100K Reviews (hotels and books): already labelled Positive / Mixed / Negative.
  * Amazon Reviews (English): 1-2 stars -> negative, 3 -> mixed, 4-5 -> positive.
  * Arabic SHEIN reviews (fashion, Gulf/Levantine dialect): kept apart as an
    out-of-domain test only. It is never used for training or tuning.

Each language gets the same number of examples per class in every split, so the
comparison between models and between languages is fair.
"""
from pathlib import Path

import pandas as pd

from src.clean import detect_language, full_clean, light_clean

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
LABELS = ["negative", "mixed", "positive"]
SEED = 42

# examples per class, per language
N_TRAIN, N_VAL, N_TEST = 5000, 1000, 1000


SOURCES = {  # Hugging Face dataset repo -> files we need
    "arbml/arabic_100k_reviews": ["data/train-00000-of-00001.parquet"],
    "SetFit/amazon_reviews_multi_en": ["train.jsonl", "validation.jsonl", "test.jsonl"],
    "Ruqiya/Arabic_Reviews_of_SHEIN": ["data/train-00000-of-00001.parquet"],
}


def download():
    """Fetch the three public datasets into data/raw/ (skips files that are already there)."""
    from huggingface_hub import hf_hub_download
    folders = {"arbml/arabic_100k_reviews": "arabic_100k_reviews", "SetFit/amazon_reviews_multi_en": "amazon_en",
               "Ruqiya/Arabic_Reviews_of_SHEIN": "Arabic_Reviews_of_SHEIN"}
    for repo, files in SOURCES.items():
        for f in files:
            if not (RAW / folders[repo] / f).exists():
                hf_hub_download(repo, f, repo_type="dataset", local_dir=RAW / folders[repo])


def stars_to_label(stars: int) -> str:
    return "negative" if stars <= 2 else ("mixed" if stars == 3 else "positive")


def load_arabic() -> pd.DataFrame:
    d = pd.read_parquet(RAW / "arabic_100k_reviews/data/train-00000-of-00001.parquet")
    d["label"] = d["label"].map({0: "positive", 1: "mixed", 2: "negative"})
    d["lang"], d["source"] = "ar", "arabic_100k (hotels, books)"
    return d[["text", "label", "lang", "source"]]


def load_english() -> pd.DataFrame:
    parts = [pd.read_json(RAW / f"amazon_en/{s}.jsonl", lines=True) for s in ["train", "validation", "test"]]
    d = pd.concat(parts, ignore_index=True)
    d["label"] = (d["label"] + 1).map(stars_to_label)   # dataset labels are 0..4 = 1..5 stars
    d["lang"], d["source"] = "en", "amazon (products)"
    return d[["text", "label", "lang", "source"]]


def load_shein() -> pd.DataFrame:
    d = pd.read_parquet(RAW / "Arabic_Reviews_of_SHEIN/data/train-00000-of-00001.parquet")
    d["label"] = d["label"].map(stars_to_label)
    d["lang"], d["source"] = "ar", "shein (fashion, dialect)"
    return d[["text", "label", "lang", "source"]]


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["text"] = d["text"].astype(str).map(light_clean)
    d["clean"] = d["text"].map(full_clean)
    d = d[d["clean"].str.split().str.len() >= 2]          # drop empty / one-word noise
    d = d.drop_duplicates("clean")                          # no duplicate can sit in two splits
    d["detected_lang"] = d["text"].map(detect_language)
    return d


def split_language(d: pd.DataFrame) -> dict:
    out = {"train": [], "val": [], "test": []}
    for label in LABELS:
        g = d[d["label"] == label].sample(N_TRAIN + N_VAL + N_TEST, random_state=SEED)
        out["train"].append(g.iloc[:N_TRAIN])
        out["val"].append(g.iloc[N_TRAIN:N_TRAIN + N_VAL])
        out["test"].append(g.iloc[N_TRAIN + N_VAL:])
    return {k: pd.concat(v) for k, v in out.items()}


def build():
    download()
    OUT.mkdir(parents=True, exist_ok=True)
    ar, en = prepare(load_arabic()), prepare(load_english())
    stats = []
    for name, d in [("ar", ar), ("en", en)]:
        stats.append({"lang": name, "rows_after_cleaning": len(d),
                      "wrong_language_rows": int((d["detected_lang"] != name).sum()),
                      **d["label"].value_counts().to_dict()})
    ar, en = ar[ar["detected_lang"] == "ar"], en[en["detected_lang"] == "en"]

    splits = {k: pd.concat([split_language(ar)[k], split_language(en)[k]]) for k in ["train", "val", "test"]}
    for k, d in splits.items():
        d = d.sample(frac=1, random_state=SEED).reset_index(drop=True)
        d.drop(columns="detected_lang").to_parquet(OUT / f"{k}.parquet", index=False)
        print(k, len(d), d.groupby(["lang", "label"]).size().to_dict())

    shein = prepare(load_shein())
    shein = shein[~shein["clean"].isin(splits["train"]["clean"])]
    shein.drop(columns="detected_lang").reset_index(drop=True).to_parquet(OUT / "shein_ood.parquet", index=False)
    print("shein out-of-domain", len(shein), shein["label"].value_counts().to_dict())

    pd.DataFrame(stats).to_csv(ROOT / "reports" / "data_stats.csv", index=False)
    print(pd.DataFrame(stats))


if __name__ == "__main__":
    build()
