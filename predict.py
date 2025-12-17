"""
predict.py

High-level prediction logic for ResIQ AI.
Uses semantic embeddings instead of TF-IDF classification.
"""

from sanity import looks_like_resume
from embedding_matcher import EmbeddingMatcher

# Similarity thresholds (tuned, explainable)
MATCH_THRESHOLD = 0.55
UNCERTAIN_MARGIN = 0.05


class ResIQPredictor:
    def __init__(self):
        self.matcher = EmbeddingMatcher()

    def predict(self, resume_text: str, job_text: str):
        """
        Returns match decision based on semantic similarity.
        """

        # Step 1: Sanity check — is this even a resume?
        if not looks_like_resume(resume_text):
            return {
                "prediction": 0,
                "confidence": 0.0,
                "uncertain": False,
                "reason": "Uploaded document does not appear to be a resume"
            }

        # Step 2: Semantic similarity
        similarity = self.matcher.similarity(resume_text, job_text)

        # Step 3: Decision logic
        prediction = int(similarity >= MATCH_THRESHOLD)
        uncertain = abs(similarity - MATCH_THRESHOLD) <= UNCERTAIN_MARGIN

        return {
            "prediction": prediction,
            "confidence": round(similarity, 3),
            "uncertain": uncertain
        }
