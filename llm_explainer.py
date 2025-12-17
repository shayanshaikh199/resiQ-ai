"""
llm_explainer.py

Optional Gemini explanation module.
Gemini is NEVER allowed to block or crash the app.
"""

import os
from dotenv import load_dotenv

load_dotenv()

ENABLE_GEMINI = os.getenv("ENABLE_GEMINI", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class GeminiExplainer:
    def __init__(self):
        self.enabled = ENABLE_GEMINI and GEMINI_API_KEY is not None

        if self.enabled:
            try:
                from google import genai
                self.client = genai.Client(api_key=GEMINI_API_KEY)
                self.model = "gemini-1.0-pro"
            except Exception as e:
                print("Gemini disabled:", e)
                self.enabled = False

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
Explain the resume–job match result and suggest improvements.

Similarity score: {similarity_score}
Decision: {decision_text}

Resume:
{resume_text[:3000]}

Job description:
{job_text[:1500]}
"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print("Gemini runtime error:", e)
            return (
                "AI explanation temporarily unavailable.\n\n"
                f"Reason: {str(e)}"
            )
