"""
Training pipeline: load data → preprocess → TF-IDF → train → evaluate → save.
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from src.preprocessor import preprocess
from src.domains import DOMAINS


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
BINARIZER_PATH = os.path.join(MODEL_DIR, "label_binarizer.joblib")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training_data.csv")


def load_data(data_path: str = DATA_PATH) -> tuple[list[str], list[list[str]]]:
    """Load and parse the training CSV. Returns (texts, label_lists)."""
    df = pd.read_csv(data_path)

    if "text" not in df.columns or "domain" not in df.columns:
        print("Error: CSV must have 'text' and 'domain' columns.")
        sys.exit(1)

    texts = df["text"].tolist()
    # Parse multi-label: "Healthcare|Machine Learning" → ["Healthcare", "Machine Learning"]
    labels = [
        [label.strip() for label in str(domains).split("|")]
        for domains in df["domain"].tolist()
    ]

    return texts, labels


def train(data_path: str = DATA_PATH, test_size: float = 0.2):
    """Full training pipeline."""
    print("Loading data...")
    texts, labels = load_data(data_path)
    print(f"  → {len(texts)} samples loaded")

    # Preprocess
    print("Preprocessing text...")
    texts_clean = [preprocess(t) for t in texts]

    # Binarize labels
    mlb = MultiLabelBinarizer(classes=DOMAINS)
    y = mlb.fit_transform(labels)
    print(f"  → {len(mlb.classes_)} domain labels")

    # Check for unseen labels in data
    all_labels = set(label for label_list in labels for label in label_list)
    unknown = all_labels - set(DOMAINS)
    if unknown:
        print(f"  ⚠ Warning: unknown labels in data (not in DOMAINS): {unknown}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts_clean, y, test_size=test_size, random_state=42
    )
    print(f"  → Train: {len(X_train)}, Test: {len(X_test)}")

    # Build pipeline
    print("Training model...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", OneVsRestClassifier(
            CalibratedClassifierCV(LinearSVC(max_iter=5000, class_weight="balanced"), cv=3)
        )),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    print("\n--- Classification Report (Test Set) ---\n")
    y_pred = pipeline.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=mlb.classes_,
        zero_division=0,
    )
    print(report)

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(mlb, BINARIZER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Label binarizer saved to {BINARIZER_PATH}")

    return pipeline, mlb


if __name__ == "__main__":
    train()
