"""
recommendations.py

Rule-based + semantic recommendation engine for ResIQ AI.
Uses vector similarity, keyword gaps, and domain signals.
No external APIs.
"""

import re
from collections import Counter
from typing import List, Dict

COMMON_SKILLS = {
    "tech": [
        "python", "java", "sql", "api", "docker", "aws", "git",
        "machine learning", "data analysis", "linux"
    ],
    "accounting": [
        "accounting", "budgeting", "financial reporting", "excel",
        "quickbooks", "tax", "auditing", "forecasting"
    ],
    "marketing": [
        "seo", "content", "social media", "branding", "analytics",
        "campaigns", "copywriting", "email marketing"
    ],
    "healthcare": [
        "patient care", "clinical", "documentation", "compliance",
        "medical", "ehr", "diagnosis"
    ],
    "general": [
        "communication", "leadership", "teamwork", "problem solving"
    ]
}


def normalize(text: str) -> List[str]:
    """Lowercase, clean, and tokenize text."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()


def detect_domain(text: str) -> str:
    """Detect dominant domain based on keyword hits."""
    tokens = normalize(text)
    scores = {}

    for domain, skills in COMMON_SKILLS.items():
        scores[domain] = sum(skill in " ".join(tokens) for skill in skills)

    return max(scores, key=scores.get)


def extract_skills(text: str) -> Counter:
    """Extract skill frequency from text."""
    tokens = normalize(text)
    joined = " ".join(tokens)

    found = []
    for skills in COMMON_SKILLS.values():
        for skill in skills:
            if skill in joined:
                found.append(skill)

    return Counter(found)


def generate_recommendations(
    resume_text: str,
    job_text: str,
    similarity_score: float,
    is_match: bool
) -> Dict[str, List[str]]:
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    missing_skills = [
        skill for skill in job_skills
        if skill not in resume_skills
    ]

    weak_skills = [
        skill for skill, count in resume_skills.items()
        if count == 1 and skill in job_skills
    ]

    resume_domain = detect_domain(resume_text)
    job_domain = detect_domain(job_text)

    explanations = []
    recommendations = []

    # Explanation
    if is_match:
        explanations.append(
            "The resume aligns with the job based on overlapping skills "
            "and semantic similarity."
        )
    else:
        explanations.append(
            "The resume does not strongly align with the job requirements."
        )

    explanations.append(
        f"Detected resume domain: {resume_domain.capitalize()}, "
        f"job domain: {job_domain.capitalize()}."
    )

    # Recommendations
    if missing_skills:
        recommendations.append(
            f"Add these missing skills to your resume: {', '.join(missing_skills[:5])}."
        )

    if weak_skills:
        recommendations.append(
            f"Strengthen these skills with examples or projects: {', '.join(weak_skills[:5])}."
        )

    if resume_domain != job_domain:
        recommendations.append(
            f"Your resume aligns more with {resume_domain.capitalize()} roles. "
            f"Consider tailoring it for {job_domain.capitalize()} positions."
        )

    if similarity_score < 0.5:
        recommendations.append(
            "Consider rewriting bullet points to better match the job language "
            "and required competencies."
        )

    if not recommendations:
        recommendations.append(
            "Your resume is well-aligned. Consider adding measurable achievements "
            "to further strengthen it."
        )

    return {
        "explanation": explanations,
        "recommendations": recommendations
    }
