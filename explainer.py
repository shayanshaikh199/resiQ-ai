"""
explainer.py

Provides human-readable explanations for why a resume
matches (or does not match) a job description.

IMPORTANT:
This module is used ONLY for explainability.
It does NOT affect the match decision.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


class MatchExplainer:
    def __init__(self):
        # TF-IDF is used ONLY to surface shared terms for explanation
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=1000
        )

    def explain(self, resume_text: str, job_text: str, top_k: int = 5):
        """
        Returns top shared keywords/phrases between resume and job.
        """
        texts = [resume_text, job_text]

        tfidf = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        resume_vec = tfidf[0].toarray()[0]
        job_vec = tfidf[1].toarray()[0]

        # Shared importance = product of TF-IDF weights
        shared_scores = resume_vec * job_vec

        # Indices of most important shared terms
        top_indices = shared_scores.argsort()[::-1][:top_k]

        keywords = [
            feature_names[i]
            for i in top_indices
            if shared_scores[i] > 0
        ]

        return keywords
