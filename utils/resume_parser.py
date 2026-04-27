"""
Utils — Resume Parser

Извлекает структурированный профиль (hard_skills, soft_skills, interests)
из произвольного текста резюме.

Режимы:
  1. Regex fallback  — ищет известные навыки из словаря KNOWN_SKILLS
  2. LLM mode        — если задан LLM_API_KEY, просит LLM распарсить резюме

Future hook: добавить PDF-чтение через pdfplumber / pypdf2
"""

from __future__ import annotations
from utils.llm_client import call_llm_json, is_available
import os
import re
import json

# ---------------------------------------------------------------------------
# Словарь известных навыков для regex-режима
# ---------------------------------------------------------------------------

KNOWN_HARD_SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C++", "C#", "PHP", "Ruby",
    "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    "Docker", "Kubernetes", "Terraform", "Ansible", "Helm",
    "AWS", "GCP", "Azure", "Linux", "Bash", "Git", "CI/CD",
    "React", "Vue", "Angular", "HTML", "CSS", "Node.js",
    "FastAPI", "Django", "Flask", "Spring", "REST API", "GraphQL",
    "Pandas", "NumPy", "Matplotlib", "Scikit-learn", "PyTorch", "TensorFlow", "Keras",
    "Jupyter", "Spark", "Airflow", "dbt", "MLflow", "MLOps",
    "System Design", "Microservices", "Kafka", "RabbitMQ",
    "Figma", "Jira", "Confluence", "Agile", "Scrum",
]

KNOWN_SOFT_SKILLS = [
    "problem solving", "teamwork", "communication", "leadership",
    "critical thinking", "time management", "adaptability", "creativity",
    "collaboration", "mentoring", "presentation", "decision making",
]

KNOWN_INTERESTS = [
    "backend", "frontend", "fullstack", "devops", "data", "AI", "ML",
    "cloud", "security", "automation", "open source", "mobile",
    "blockchain", "embedded", "game development",
]


def _extract_by_regex(text: str) -> dict:
    """Простой regex-парсер: ищет упоминания известных навыков в тексте."""
    text_lower = text.lower()
    hard  = [s for s in KNOWN_HARD_SKILLS  if s.lower() in text_lower]
    soft  = [s for s in KNOWN_SOFT_SKILLS  if s.lower() in text_lower]
    inter = [s for s in KNOWN_INTERESTS    if s.lower() in text_lower]

    # Попытка извлечь имя (первая строка или после "Name:")
    name = "User"
    name_match = re.search(r"(?:name[:\s]+)([A-Z][a-z]+(?: [A-Z][a-z]+)?)", text, re.I)
    if name_match:
        name = name_match.group(1).strip()
    else:
        first_line = text.strip().split("\n")[0].strip()
        if len(first_line.split()) <= 4 and first_line[0].isupper():
            name = first_line

    return {
        "name":        name,
        "hard_skills": hard,
        "soft_skills": soft,
        "interests":   inter,
        "_meta":       {"source": "resume_parser_regex"},
    }


def _extract_by_llm(text: str) -> dict:
    """LLM-парсер: просит модель вернуть структурированный JSON."""
    if not is_available():
        raise ValueError("LLM_API_KEY не задан")

    prompt = f"""
Ты — парсер резюме. Прочитай текст ниже и верни ТОЛЬКО валидный JSON:
{{
  "name": "имя кандидата",
  "hard_skills": ["список технических навыков"],
  "soft_skills": ["список гибких навыков"],
  "interests": ["профессиональные интересы / области"]
}}

Текст резюме:
\"\"\"
{text[:3000]}
\"\"\"

Только JSON, без пояснений.
""".strip()

    data = call_llm_json(prompt, max_tokens=1024)
    data["_meta"] = {"source": "resume_parser_llm", "provider": provider}
    return data


def parse_resume(text: str, use_llm: bool = False) -> dict:
    """
    Главная функция парсера резюме.

    Args:
        text:    текст резюме (plain text)
        use_llm: принудительно использовать LLM

    Returns:
        dict с ключами: name, hard_skills, soft_skills, interests, _meta
    """
    if not text or not text.strip():
        return {
            "name": "User", "hard_skills": [], "soft_skills": [],
            "interests": [], "_meta": {"source": "empty"},
        }

    # Пробуем LLM если доступен
    if use_llm or os.getenv("LLM_API_KEY"):
        try:
            return _extract_by_llm(text)
        except Exception as e:
            print(f"[ResumeParser] LLM недоступен ({e}), переключаюсь на regex.")

    return _extract_by_regex(text)


# ---------------------------------------------------------------------------
# CLI тест
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = """
    John Doe
    Backend Developer

    Skills: Python, FastAPI, PostgreSQL, Docker, Git, Redis, AWS
    Soft skills: problem solving, teamwork, communication
    Interests: backend, cloud, automation
    """
    result = parse_resume(sample)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
