"""
recommendations.py

Requirement-based resume evaluation (correct approach).
"""

from requirements_extractor import extract_requirements
from coverage import requirement_covered


def generate_recommendations(
    resume_text: str,
    job_text: str,
    similarity_score: float,
    is_match: bool,
    vectorizer
) -> dict:

    explanation = []
    issues = []
    recommendations = []

    # -------------------------
    # Explanation
    # -------------------------
    explanation.append(
        f"Semantic similarity score: {round(similarity_score, 3)}"
    )

    if is_match:
        explanation.append(
            "The resume generally aligns with the role."
        )
    else:
        explanation.append(
            "The resume does not strongly align with the role."
        )

    # -------------------------
    # Requirement extraction
    # -------------------------
    requirements = extract_requirements(job_text)

    missing_requirements = []

    for req in requirements:
        if not requirement_covered(req, resume_text, vectorizer):
            missing_requirements.append(req)

    # -------------------------
    # Issues & Recommendations
    # -------------------------
    if missing_requirements:
        issues.append(
            "Some explicit job requirements are not clearly addressed in the resume."
        )

        for req in missing_requirements[:5]:
            recommendations.append(
                f"Add or strengthen evidence for: {req}"
            )

    else:
        recommendations.append(
            "All explicit job requirements appear to be covered."
        )

    return {
        "explanation": explanation,
        "issues": issues,
        "recommendations": recommendations
    }
