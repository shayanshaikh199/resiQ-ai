"""
predict.py

Final prediction + explanation pipeline for ResIQ AI.
"""

from sanity import looks_like_resume
from embedding_matcher import EmbeddingMatcher
from llm_explainer import GeminiExplainer

MATCH_THRESHOLD = 0.55
UNCERTAIN_MARGIN = 0.05


class ResIQPredictor:
    def __init__(self):
        self.matcher = EmbeddingMatcher()
        self.llm = GeminiExplainer()

    def predict(self, resume_text: str, job_text: str):
        # Step 1: Sanity check
        if not looks_like_resume(resume_text):
            return {
                "prediction": 0,
                "confidence": 0.0,
                "uncertain": False,
                "reason": "Uploaded document does not appear to be a resume",
                "ai_explanation": None
            }

        # Step 2: Semantic similarity
        similarity = self.matcher.similarity(resume_text, job_text)
        prediction = int(similarity >= MATCH_THRESHOLD)
        uncertain = abs(similarity - MATCH_THRESHOLD) <= UNCERTAIN_MARGIN

        # Step 3: Gemini explanation (UX layer only)
        explanation = self.llm.explain(
            resume_text=resume_text,
            job_text=job_text,
            similarity_score=round(similarity, 3),
            is_match=prediction == 1
        )

        return {
            "prediction": prediction,
            "confidence": round(similarity, 3),
            "uncertain": uncertain,
            "ai_explanation": explanation
        }
