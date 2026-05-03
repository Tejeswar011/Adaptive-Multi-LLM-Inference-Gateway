import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import class_weight
import numpy as np
import joblib
import os

LOG_PATH = "training/router_logs.csv"
MODEL_PATH = "routing/trained_router.pkl"
MIN_ROWS = 20


def train():
    if not os.path.isfile(LOG_PATH):
        print(f"No training data found at {LOG_PATH}")
        return False

    df = pd.read_csv(LOG_PATH)

    if len(df) < MIN_ROWS:
        print(f"Not enough data: {len(df)} rows (need {MIN_ROWS})")
        return False

    X = df[["token_length", "word_count", "question_type", "load_score"]]
    y = df["selected_tier"]

    classes = y.unique()
    if len(classes) < 2:
        print(f"Only 1 class in data ({classes[0]}), need at least 2 tiers")
        return False

    # Compute class weights to handle imbalanced tiers
    weights = class_weight.compute_sample_weight("balanced", y)

    model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2)
    model.fit(X, y, sample_weight=weights)

    # Cross-validation (if enough data)
    if len(df) >= 30:
        scores = cross_val_score(model, X, y, cv=min(5, len(df) // 5), scoring="accuracy")
        print(f"Cross-val accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Retrain on full data for deployment
    model.fit(X, y, sample_weight=weights)
    joblib.dump(model, MODEL_PATH)

    print(f"Router trained on {len(df)} rows, {len(classes)} classes: {list(classes)}")
    print(f"Model saved to {MODEL_PATH}")
    return True


if __name__ == "__main__":
    train()
