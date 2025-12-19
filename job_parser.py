"""
job_parser.py

Extracts explicit job requirements using WORD-LEVEL matching.
"""

import re
from typing import List


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))


def extract_required_skills(text: str) -> List[str]:
    WORD_SKILLS = {
        "python", "java", "sql", "excel", "git", "aws",
        "seo", "content", "analytics", "marketing",
        "accounting", "finance", "communication", "leadership"
    }

    tokens = tokenize(text)
    return sorted(list(WORD_SKILLS.intersection(tokens)))


def extract_required_degree(text: str) -> str | None:
    text_lower = text.lower()

    if re.search(r"computer science|b\.?sc|cs degree", text_lower):
        return "Computer Science"
    if re.search(r"marketing degree|bachelor of marketing", text_lower):
        return "Marketing"
    if re.search(r"accounting degree|cpa|bachelor of accounting", text_lower):
        return "Accounting"
    if re.search(r"engineering degree|bachelor of engineering", text_lower):
        return "Engineering"

    return None


def extract_required_experience(text: str) -> int:
    match = re.search(r"(\d+)\+?\s+years", text.lower())
    return int(match.group(1)) if match else 0
