"""
llm_explainer.py

Gemini explanation module using Google AI Studio REST API.
Compatible with free AI Studio API keys.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ENABLE_GEMINI = os.getenv("ENABLE_GEMINI", "false").lower() == "true"

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-1.5-flash:generateContent"
)


class GeminiExplainer:
    def __init__(self):
        self.enabled = ENABLE_GEMINI and GEMINI_API_KEY is not None

    def explain(
        self,
        resume_text: str,
        job_text: str,
        similarity_score: float,
        is_match: bool
    ) -> str:
        if not self.enabled:
            return (
                "AI explanation is currently disabled.\n\n"
                "Reason: External LLM calls are turned off for stability."
            )

        decision_text = "Match" if is_match else "No Match"

        prompt = f"""
You are an expert career coach and recruiter.

A resume was compared to a job description using an AI matching system.

Similarity score: {similarity_score}
Final decision: {decision_text}

Explain why this result makes sense.
Then suggest 3 concrete improvements to better match the job.

Resume:
{resume_text[:2500]}

Job Description:
{job_text[:1500]}
"""

        try:
            response = requests.post(
                GEMINI_ENDPOINT,
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [
                        {
                            "parts": [{"text": prompt}]
                        }
                    ]
                },
                timeout=15
            )

            response.raise_for_status()
            data = response.json()

            return data["candidates"][0]["content"]["parts"][0]["text"]

        except Exception as e:
            print("Gemini REST error:", e)
            return (
                "AI explanation temporarily unavailable.\n\n"
                f"Reason: {str(e)}"
            )
