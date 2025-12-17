from io import BytesIO
from pdfminer.high_level import extract_text


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Safely extract text from a PDF file.
    Returns empty string if extraction fails.
    """
    try:
        with BytesIO(pdf_bytes) as pdf_stream:
            text = extract_text(pdf_stream)
            return text.strip() if text else ""
    except Exception as e:
        print("PDF extraction error:", e)
        return ""
