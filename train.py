import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/training_data.csv"
MODEL_PATH = "models/resiq_model.joblib"


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # 🔑 CRITICAL: combine resume + job text
    X = (df["resume_text"] + " " + df["job_text"]).values
    y = df["label"].values

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    # Train
    pipeline.fit(X, y)

    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print("Model trained and saved to", MODEL_PATH)


if __name__ == "__main__":
    main()
