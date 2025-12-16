import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

DATA_PATH = os.path.join("data", "seed_pairs.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "resiq_model.joblib")


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Combine resume + job text
    X = (df["resume_text"] + " " + df["job_text"]).values
    y = df["label"].values

    # ML pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train
    pipeline.fit(X, y)

    # Evaluate (on same data for now — prototype stage)
    preds = pipeline.predict(X)
    print("Training Evaluation:")
    print(classification_report(y, preds))

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
