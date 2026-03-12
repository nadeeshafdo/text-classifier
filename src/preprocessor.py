"""
Text preprocessing pipeline for the classifier.

Cleans and normalizes raw text input before vectorization.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _ensure_nltk_data():
    """Download required NLTK data if not present."""
    for resource in ["punkt_tab", "stopwords", "wordnet"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def preprocess(text: str) -> str:
    """
    Clean and normalize text for classification.

    Pipeline:
        1. Lowercase
        2. Strip URLs, emails, special chars
        3. Tokenize
        4. Remove stopwords
        5. Lemmatize
        6. Rejoin
    """
    if not text or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove emails
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Remove special characters and digits (keep letters and spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    tokens = [
        _lemmatizer.lemmatize(token)
        for token in tokens
        if token not in _stop_words and len(token) > 2
    ]

    return " ".join(tokens)
