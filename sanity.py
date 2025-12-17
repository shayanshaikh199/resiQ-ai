"""
sanity.py

Lightweight sanity check to verify input resembles a resume.
This should NEVER be overly strict.
"""

def looks_like_resume(text: str) -> bool:
    if not text:
        return False

    text = text.lower()

    # Very common resume signals
    resume_indicators = [
        "experience",
        "education",
        "skills",
        "work",
        "projects",
        "summary",
        "certification",
        "employment",
    ]

    # Length check (PDFs often have noise, so keep low)
    if len(text) < 300:
        return False

    matches = sum(1 for kw in resume_indicators if kw in text)

    # ✅ Only require ONE signal
    return matches >= 1
