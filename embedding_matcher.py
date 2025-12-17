"""
embedding_matcher.py

This module handles semantic matching between a resume and a job description
using sentence embeddings and cosine similarity.

Why embeddings?
- Captures semantic meaning, not just word overlap
- Fixes cross-domain false matches (marketing ≠ accounting)
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class EmbeddingMatcher:
    def __init__(self):
        """
        Load the sentence embedding model lazily.
        This prevents blocking app startup.
        """
        self.model = None

    def load_model(self):
        if self.model is None:
            print("DEBUG: Loading embedding model...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            print("DEBUG: Embedding model loaded")

    def similarity(self, resume_text: str, job_text: str) -> float:
        """
        Compute cosine similarity between resume and job description.
        Returns a value between 0 and 1.
        """
        self.load_model()

        # Encode both texts into dense vectors
        embeddings = self.model.encode(
            [resume_text, job_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Compute cosine similarity
        score = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]

        return float(score)
