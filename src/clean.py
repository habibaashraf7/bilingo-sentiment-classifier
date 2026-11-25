"""Text cleaning for Arabic and English customer feedback.

Two levels:
  * light_clean: what the transformer sees. Removes URLs, HTML and extra spaces only,
    because a pretrained tokenizer already knows punctuation, casing and emojis.
  * full_clean: what the TF-IDF baseline sees. Also normalises Arabic spelling
    variants, lowercases English, and squeezes repeated letters, so that
    "رااائع" / "رائع" or "Sooo good" / "so good" become the same token.

Stopwords are NOT removed: words like "not", "but", "لا", "لكن" carry sentiment.
"""
import re
import unicodedata

ARABIC_LETTERS = re.compile(r"[ء-ي]")
LATIN_LETTERS = re.compile(r"[A-Za-z]")
DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
URL = re.compile(r"https?://\S+|www\.\S+")
HTML = re.compile(r"<[^>]+>")
MENTION = re.compile(r"@\w+")
REPEATS = re.compile(r"(.)\1{2,}")
SPACES = re.compile(r"\s+")
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
ARABIC_PUNCT = str.maketrans({"،": ",", "؛": ";", "؟": "?"})


def detect_language(text: str) -> str:
    """'ar' if most letters are Arabic, otherwise 'en'."""
    ar = len(ARABIC_LETTERS.findall(text))
    en = len(LATIN_LETTERS.findall(text))
    return "ar" if ar >= en else "en"


def light_clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = HTML.sub(" ", text)
    text = URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def normalize_arabic(text: str) -> str:
    text = DIACRITICS.sub("", text).replace(TATWEEL, "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    return text


def full_clean(text: str) -> str:
    text = light_clean(text)
    text = text.translate(ARABIC_DIGITS).translate(ARABIC_PUNCT)
    text = normalize_arabic(text)
    text = text.lower()
    text = re.sub(r"n't\b", " not", text)          # don't -> do not (keeps the negation as a token)
    text = REPEATS.sub(r"\1\1", text)               # soooo -> soo, رااائع -> راائع
    text = re.sub(r"[^\w\s!?؀-ۿ☀-➿\U0001F300-\U0001FAFF]", " ", text)  # keep ! ? and emojis
    text = re.sub(r"([!?☀-➿\U0001F300-\U0001FAFF])", r" \1 ", text)              # each emoji is its own token
    return SPACES.sub(" ", text).strip()


if __name__ == "__main__":
    for s in ["الفندق رااائع جداً 😍 والموظفين متعاونين!!!", "I didn't like it... Sooo disappointing http://x.co"]:
        print(detect_language(s), "|", light_clean(s), "|", full_clean(s))
