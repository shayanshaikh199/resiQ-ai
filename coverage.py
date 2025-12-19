"""
coverage.py

Checks whether job requirements are clearly covered by resume content.
"""

from sklearn.metrics.pairwise import cosine_similarity


def requirement_covered(requirement: str, resume_text: str, vectorizer) -> bool:
    vectors = vectorizer.transform([requirement, resume_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    # Conservative threshold
    return similarity >= 0.55
