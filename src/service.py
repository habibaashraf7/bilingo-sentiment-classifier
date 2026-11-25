"""Small Python API used by the app, the inference script and the evaluation.

    from src.service import SentimentService
    s = SentimentService()
    s.predict(["الخدمة ممتازة", "Arrived broken"], model="transformer")
"""
from functools import cached_property
from pathlib import Path

import joblib
import numpy as np

from src.clean import detect_language, full_clean, light_clean

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["negative", "mixed", "positive"]
TRANSFORMER_DIR = ROOT / "models" / "xlmr-sentiment"


class SentimentService:
    @cached_property
    def baseline(self):
        return joblib.load(ROOT / "models" / "baseline_tfidf.joblib")

    _transformer_failed = False

    @property
    def has_transformer(self) -> bool:
        return (TRANSFORMER_DIR / "config.json").exists() and not self._transformer_failed

    def warm_up(self):
        """Load the transformer once, up front. If it can't load, fall back to the baseline instead of crashing."""
        if not self.has_transformer:
            return False
        try:
            self._transformer
            return True
        except Exception as e:  # e.g. missing torch / transformers
            print(f"Transformer unavailable, using the baseline: {type(e).__name__}: {e}")
            self._transformer_failed = True
            return False

    @cached_property
    def _transformer(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return tok, model.to(device), device

    def baseline_proba(self, texts):
        clf = self.baseline
        p = clf.predict_proba([full_clean(t) for t in texts])
        order = [list(clf.classes_).index(l) for l in LABELS]
        return p[:, order]

    def transformer_proba(self, texts, batch_size=64):
        import torch
        tok, model, device = self._transformer
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [light_clean(t) for t in texts[i:i + batch_size]]
            enc = tok(batch, truncation=True, max_length=192, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
        return np.vstack(out)

    def predict(self, texts, model="transformer"):
        texts = list(texts)
        if model == "transformer" and not self.has_transformer:
            model = "baseline"
        p = self.transformer_proba(texts) if model == "transformer" else self.baseline_proba(texts)
        return [{"text": t, "language": detect_language(t), "label": LABELS[int(r.argmax())],
                 "confidence": float(r.max()), "scores": dict(zip(LABELS, map(float, r))), "model": model}
                for t, r in zip(texts, p)]

    def explain_baseline(self, text, top=6):
        """Words / character pieces that pushed the baseline towards its predicted class."""
        clf = self.baseline
        union, lr = clf.named_steps["tfidf"], clf.named_steps["clf"]
        x = union.transform([full_clean(text)])
        names = np.concatenate([[f"{n}:{f}" for f in v.get_feature_names_out()] for n, v in union.transformer_list])
        label = clf.predict([full_clean(text)])[0]
        k = list(lr.classes_).index(label)
        contrib = x.multiply(lr.coef_[k]).tocsr()
        idx = contrib.indices[np.argsort(-contrib.data)]
        words = [names[i].split(":", 1)[1] for i in idx if names[i].startswith("word:")]
        words = [w for w in words if not set(w.split()) <= DISPLAY_STOPWORDS][:top]
        return label, words


# only used to hide filler words in the explanation shown to users; the model itself keeps them
DISPLAY_STOPWORDS = set("the a an and or to of in on at for with it this that is was were be been i we you they "
                        "my our your its me us he she his her as by from are am so has have had do".split()) | \
                    set("في من على الى عن و او ان هذا هذه كان كانت مع هو هي انا ال".split())
