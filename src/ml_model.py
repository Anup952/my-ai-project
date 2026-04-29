"""Machine learning pipeline helpers for supervised sentiment classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"


def clean_labeled_dataset(df: pd.DataFrame, text_column: str, label_column: str) -> pd.DataFrame:
    """Return rows with valid text and label values."""
    cleaned = df[[text_column, label_column]].copy()
    cleaned.columns = ["text", "label"]
    cleaned["text"] = cleaned["text"].fillna("").astype(str).str.strip()
    cleaned["label"] = cleaned["label"].fillna("").astype(str).str.strip().str.title()
    cleaned = cleaned[(cleaned["text"] != "") & (cleaned["label"] != "")]
    return cleaned


def build_sentiment_pipeline() -> Pipeline:
    """Create a TF-IDF + Logistic Regression sentiment classification pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=12000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_sentiment_model(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Train, evaluate, and persist a supervised sentiment model."""
    cleaned = clean_labeled_dataset(df, text_column, label_column)
    if len(cleaned) < 6:
        raise ValueError("Dataset must contain at least 6 valid labeled rows.")
    if cleaned["label"].nunique() < 2:
        raise ValueError("Dataset must contain at least 2 sentiment classes.")

    stratify = cleaned["label"] if cleaned["label"].value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        cleaned["text"],
        cleaned["label"],
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    model = build_sentiment_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = sorted(cleaned["label"].unique())

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return {
        "model": model,
        "model_path": MODEL_PATH,
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels),
        "labels": labels,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "class_distribution": cleaned["label"].value_counts().reset_index(),
    }


def model_exists() -> bool:
    """Return whether a trained model artifact exists."""
    return MODEL_PATH.exists()


def load_model() -> Pipeline:
    """Load the persisted trained model."""
    if not model_exists():
        raise FileNotFoundError("No trained model found. Train a model first.")
    return joblib.load(MODEL_PATH)


def predict_with_model(text: str) -> tuple[str, float | None]:
    """Predict sentiment using the saved supervised model."""
    model = load_model()
    prediction = model.predict([text])[0]
    confidence = None
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        confidence = float(model.predict_proba([text]).max())
    return prediction, confidence

