"""
requirements_extractor.py

Extracts explicit job requirements from a job description.
Conservative by design.
"""

import re

ACTION_VERBS = [
    "experience", "ability", "knowledge", "familiarity",
    "proficiency", "skill", "skills", "responsible",
    "responsibilities", "required", "requirements"
]


def extract_requirements(job_text: str) -> list[str]:
    requirements = []

    lines = job_text.split("\n")

    for line in lines:
        clean = line.strip().lower()

        if len(clean) < 10:
            continue

        # Bullet points or requirement-like sentences
        if clean.startswith(("-", "•")) or any(v in clean for v in ACTION_VERBS):
            # Remove bullet characters
            req = re.sub(r"^[•\-–]+", "", clean).strip()

            # Avoid vague lines
            if len(req.split()) >= 4:
                requirements.append(req.capitalize())

    return list(set(requirements))
