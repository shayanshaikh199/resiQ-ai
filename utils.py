"""
utils.py

Robust PDF text extraction for FastAPI uploads.
"""

from PyPDF2 import PdfReader
import io


def extract_text_from_pdf(uploaded_file) -> str:
    try:
        # Read file bytes once
        pdf_bytes = uploaded_file.file.read()

        # Wrap bytes in a buffer PyPDF2 understands
        pdf_stream = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_stream)

        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

        return "\n".join(text) if text else "No readable text found in PDF."

    except Exception as e:
        print("PDF extraction error:", e)
        raise RuntimeError("Failed to extract text from PDF")
