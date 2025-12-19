"""
generate_dataset.py

Generates a realistic, domain-agnostic dataset of resume–job pairs
for training ResIQ AI.

Uses hard negatives to reduce false positives.
"""

import random
import csv
import os

OUTPUT_PATH = "data/training_data.csv"
NUM_POSITIVE = 250
NUM_NEGATIVE = 250

random.seed(42)

# ---------- Core skill themes (not domains) ----------
SKILL_THEMES = [
    # Education / Creative
    ["lesson planning", "instruction", "feedback", "assessment", "curriculum"],
    ["choreography", "performance", "rehearsal", "creative direction"],

    # Business / Operations
    ["budgeting", "forecasting", "financial reporting", "analysis"],
    ["inventory management", "logistics", "scheduling", "process improvement"],

    # Tech / Data
    ["python", "data analysis", "automation", "api integration"],
    ["software development", "testing", "debugging", "version control"],

    # Marketing / Communication
    ["content creation", "campaign management", "analytics", "branding"],
    ["client communication", "stakeholder management", "presentation"],

    # Healthcare / Service
    ["patient care", "documentation", "compliance", "coordination"],
    ["case management", "support services", "record keeping"]
]

ACTION_PHRASES = [
    "led", "assisted with", "coordinated", "developed",
    "implemented", "supported", "managed", "designed"
]

OUTCOMES = [
    "to improve efficiency",
    "to support team goals",
    "to enhance quality",
    "to meet organizational objectives",
    "to increase consistency"
]


def build_paragraph(skills):
    actions = random.sample(ACTION_PHRASES, 2)
    selected_skills = random.sample(skills, min(3, len(skills)))
    outcome = random.choice(OUTCOMES)

    sentences = [
        f"{actions[0].capitalize()} {selected_skills[0]} and {selected_skills[1]} {outcome}.",
        f"{actions[1].capitalize()} {selected_skills[-1]} while collaborating with others."
    ]

    return " ".join(sentences)


def generate_positive_pair():
    theme = random.choice(SKILL_THEMES)
    resume = build_paragraph(theme)
    job = build_paragraph(theme)
    return resume, job, 1


def generate_hard_negative_pair():
    theme_resume = random.choice(SKILL_THEMES)
    theme_job = random.choice([t for t in SKILL_THEMES if t != theme_resume])

    resume = build_paragraph(theme_resume)

    # Make job *sound* similar structurally but with different skills
    job = build_paragraph(theme_job)

    return resume, job, 0


def main():
    os.makedirs("data", exist_ok=True)

    rows = []

    for _ in range(NUM_POSITIVE):
        rows.append(generate_positive_pair())

    for _ in range(NUM_NEGATIVE):
        rows.append(generate_hard_negative_pair())

    random.shuffle(rows)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["resume_text", "job_text", "label"])
        writer.writerows(rows)

    print(f"Dataset generated: {OUTPUT_PATH}")
    print(f"Total samples: {len(rows)}")


if __name__ == "__main__":
    main()
