import random
import pandas as pd

# ----------------------------
# Phrase variations
# ----------------------------

INTRO_PHRASES = [
    "Student with experience in",
    "Professional with background in",
    "Individual skilled in",
    "Candidate experienced in",
    "Graduate with training in"
]

SECTION_TITLES = [
    "Skills",
    "Experience",
    "Background",
    "Profile",
    "Summary",
    "Strengths"
]

# ----------------------------
# Domain-specific skills
# ----------------------------

DOMAINS = {
    "tech": {
        "skills": [
            "Python", "software development", "debugging",
            "data analysis", "APIs", "problem solving"
        ],
        "jobs": [
            "Software internship involving development and debugging.",
            "Technology role requiring programming and system design."
        ]
    },
    "accounting": {
        "skills": [
            "bookkeeping", "financial statements", "Excel",
            "auditing", "tax preparation", "cost accounting"
        ],
        "jobs": [
            "Accounting internship focused on financial reporting.",
            "Finance role involving bookkeeping and analysis."
        ]
    },
    "marketing": {
        "skills": [
            "social media marketing", "content creation",
            "brand strategy", "market research", "campaign analysis"
        ],
        "jobs": [
            "Marketing internship focused on campaigns and branding.",
            "Role involving customer engagement and content creation."
        ]
    },
    "healthcare": {
        "skills": [
            "patient care", "medical records",
            "clinical support", "healthcare administration"
        ],
        "jobs": [
            "Healthcare support role assisting patients.",
            "Clinical assistant position in a healthcare setting."
        ]
    },
    "dance": {
        "skills": [
            "choreography", "stage performance",
            "rehearsals", "ballet", "contemporary dance"
        ],
        "jobs": [
            "Dance performer role requiring stage experience.",
            "Creative position focused on choreography and performance."
        ]
    }
}

# ----------------------------
# Resume generator
# ----------------------------

def generate_resume(domain):
    intro = random.choice(INTRO_PHRASES)
    section = random.choice(SECTION_TITLES)
    skills = random.sample(DOMAINS[domain]["skills"], k=3)

    resume = f"""
    {intro} {domain}-related work.

    {section}:
    - {skills[0]}
    - {skills[1]}
    - {skills[2]}

    Education:
    Relevant coursework and practical experience.
    """
    return resume.strip()


def generate_job(domain):
    return random.choice(DOMAINS[domain]["jobs"])


# ----------------------------
# Dataset generation
# ----------------------------

def generate_dataset(n_samples=800):
    rows = []
    domain_list = list(DOMAINS.keys())

    # Positive examples
    for _ in range(n_samples // 2):
        domain = random.choice(domain_list)
        resume = generate_resume(domain)
        job = generate_job(domain)
        rows.append([resume, job, 1])

    # Negative examples
    for _ in range(n_samples // 2):
        resume_domain, job_domain = random.sample(domain_list, 2)
        resume = generate_resume(resume_domain)
        job = generate_job(job_domain)
        rows.append([resume, job, 0])

    random.shuffle(rows)
    return pd.DataFrame(rows, columns=["resume_text", "job_text", "label"])


if __name__ == "__main__":
    df = generate_dataset(800)  # 🔁 change to 500 or 1000 if you want
    df.to_csv("data/training_data.csv", index=False)
    print("Generated dataset with", len(df), "samples")
