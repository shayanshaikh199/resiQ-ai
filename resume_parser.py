"""
resume_parser.py

Extracts structured information from resumes:
education, skills, experience, certifications.
"""

import re
from typing import Dict, List


SECTION_HEADERS = {
    "education": ["education", "academic background", "degrees"],
    "skills": ["skills", "strengths", "technical skills", "core competencies"],
    "experience": ["experience", "work experience", "employment"],
    "certifications": ["certifications", "licenses"]
}


def split_sections(text: str) -> Dict[str, str]:
    text_lower = text.lower()
    sections = {key: "" for key in SECTION_HEADERS}

    current_section = None
    for line in text_lower.splitlines():
        for section, headers in SECTION_HEADERS.items():
            if any(h in line for h in headers):
                current_section = section
                break
        if current_section:
            sections[current_section] += line + "\n"

    return sections


def extract_degrees(text: str) -> List[str]:
    DEGREE_PATTERNS = [
        r"bachelor of [a-z\s]+",
        r"master of [a-z\s]+",
        r"b\.sc",
        r"m\.sc",
        r"computer science",
        r"engineering",
        r"business administration"
    ]
    found = []
    for pattern in DEGREE_PATTERNS:
        if re.search(pattern, text.lower()):
            found.append(pattern.replace(r"\.", "").title())
    return list(set(found))


def extract_years_experience(text: str) -> int:
    match = re.search(r"(\d+)\+?\s+years", text.lower())
    return int(match.group(1)) if match else 0
