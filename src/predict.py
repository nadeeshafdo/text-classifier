"""
Prediction module: load trained model and classify text with confidence scores.
"""

import os

import joblib

from src.preprocessor import preprocess
from src.domains import DOMAINS


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
BINARIZER_PATH = os.path.join(MODEL_DIR, "label_binarizer.joblib")

# Module-level cache
_pipeline = None
_mlb = None


def _load_model():
    """Load model and binarizer from disk (cached)."""
    global _pipeline, _mlb

    if _pipeline is not None and _mlb is not None:
        return _pipeline, _mlb

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run 'python main.py train' first."
        )

    _pipeline = joblib.load(MODEL_PATH)
    _mlb = joblib.load(BINARIZER_PATH)
    return _pipeline, _mlb


def predict(text: str) -> list[dict]:
    """
    Classify a text paragraph and return ALL domains with confidence scores.

    Returns:
        Sorted list of dicts: [{"domain": str, "confidence": float}, ...]
        Sorted by confidence descending. All 30 domains are always returned.
        Application side decides the acceptance threshold.
    """
    pipeline, mlb = _load_model()

    # Preprocess
    clean_text = preprocess(text)

    if not clean_text:
        return [{"domain": d, "confidence": 0.0} for d in DOMAINS]

    # Get probabilities for all domains
    probabilities = pipeline.predict_proba([clean_text])[0]

    # Build results
    results = [
        {"domain": domain, "confidence": round(float(prob), 4)}
        for domain, prob in zip(mlb.classes_, probabilities)
    ]

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)

    return results
