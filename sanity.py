RESUME_STRUCTURE_KEYWORDS = {
    "education",
    "experience",
    "skills",
    "projects",
    "work",
    "profile",
    "summary",
    "strengths"
}


def looks_like_resume(text: str) -> bool:
    """
    Returns True if text looks like a resume
    (filters out essays / random assignments).
    """
    text = text.lower()
    hits = sum(1 for k in RESUME_STRUCTURE_KEYWORDS if k in text)
    return hits >= 2
