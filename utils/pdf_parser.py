"""
Utils — PDF Resume Parser

Извлекает текст из PDF-файла резюме и передаёт в resume_parser.py.

Библиотеки (по приоритету):
  1. pdfplumber  — лучшее качество, сохраняет структуру
  2. pypdf       — fallback, менее точный для таблиц
  3. pdfminer    — ещё один fallback

Установка:
  pip install pdfplumber          # рекомендуется
  pip install pypdf               # fallback
"""

from __future__ import annotations
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Извлечение текста из PDF
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Извлекает весь текст из PDF-файла.
    Пробует библиотеки в порядке приоритета.

    Returns:
        str — полный текст резюме
    Raises:
        RuntimeError — если ни одна библиотека не установлена
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF не найден: {pdf_path}")

    # Попытка 1: pdfplumber (лучшее качество)
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        text = "\n".join(text_parts)
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        print(f"[PDFParser] pdfplumber ошибка: {e}")

    # Попытка 2: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        text_parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
        text = "\n".join(text_parts)
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        print(f"[PDFParser] pypdf ошибка: {e}")

    # Попытка 3: pdfminer
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(str(pdf_path))
        if text and text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        print(f"[PDFParser] pdfminer ошибка: {e}")

    raise RuntimeError(
        "Ни одна PDF-библиотека не установлена.\n"
        "Установи: pip install pdfplumber"
    )


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Извлекает текст из PDF, переданного как bytes (для Streamlit file_uploader).
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        return extract_text_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Полный pipeline: PDF → структурированный профиль
# ---------------------------------------------------------------------------

def parse_pdf_resume(
    pdf_source: str | Path | bytes,
    use_llm: bool = False,
) -> dict:
    """
    Полный pipeline: PDF файл/bytes → структурированный профиль.

    Args:
        pdf_source: путь к PDF, Path или bytes (из file_uploader)
        use_llm:    использовать LLM для извлечения навыков

    Returns:
        dict с ключами: name, hard_skills, soft_skills, interests, _meta
    """
    from utils.resume_parser import parse_resume

    # Извлекаем текст
    if isinstance(pdf_source, bytes):
        raw_text = extract_text_from_bytes(pdf_source)
    else:
        raw_text = extract_text_from_pdf(pdf_source)

    print(f"[PDFParser] Извлечено {len(raw_text)} символов из PDF")

    # Парсим текст через resume_parser
    profile = parse_resume(raw_text, use_llm=use_llm)
    profile["_meta"]["pdf_chars"] = len(raw_text)
    profile["_meta"]["source"]    = profile["_meta"].get("source","") + "+pdf"

    return profile


def get_available_backend() -> str:
    """Возвращает название доступной PDF-библиотеки."""
    try:
        import pdfplumber
        return f"pdfplumber {pdfplumber.__version__}"
    except ImportError:
        pass
    try:
        import pypdf
        return f"pypdf {pypdf.__version__}"
    except ImportError:
        pass
    try:
        import pdfminer
        return "pdfminer"
    except ImportError:
        pass
    return "не установлена"


# ---------------------------------------------------------------------------
# CLI тест
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    backend = get_available_backend()
    print(f"PDF backend: {backend}")

    if len(sys.argv) > 1:
        import json
        profile = parse_pdf_resume(sys.argv[1])
        print(json.dumps(
            {k: v for k, v in profile.items() if not k.startswith("_")},
            indent=2, ensure_ascii=False
        ))
    else:
        print("Использование: python utils/pdf_parser.py path/to/resume.pdf")
