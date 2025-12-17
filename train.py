"""
train.py

Trains the TF-IDF vectorizer and similarity model for ResIQ AI.
Saves both artifacts to the models/ directory.
"""

import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/training_data.csv"
MODEL_DIR = "models"


def main():
    # Ensure models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load training data
    df = pd.read_csv(DATA_PATH)

    # Combine resume + job text for vectorizer training
    corpus = (
        df["resume_text"].astype(str).tolist()
        + df["job_text"].astype(str).tolist()
    )

    # Train TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000
    )
    vectorizer.fit(corpus)

    # Dummy similarity model placeholder
    # (Similarity is computed dynamically in predict.py)
    model = {"type": "cosine_similarity"}

    # Save artifacts
    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.joblib")
    joblib.dump(model, f"{MODEL_DIR}/resiq_model.joblib")

    print("Training complete.")
    print("Saved vectorizer to models/vectorizer.joblib")
    print("Saved model to models/resiq_model.joblib")


if __name__ == "__main__":
    main()
