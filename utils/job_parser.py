"""
Utils — Job Description Parser

Парсит текст вакансии и возвращает:
  - required_skills  (обязательные навыки)
  - nice_to_have     (желательные)
  - role_name        (название роли)
  - seniority        (junior / middle / senior)
  - gap_analysis     (сравнение с профилем пользователя)

Режимы:
  1. Regex fallback — ищет навыки из словаря + паттерны сениорности
  2. LLM mode       — Claude/GPT парсит вакансию структурированно
"""

from __future__ import annotations
from utils.llm_client import call_llm_json, is_available
import os
import re
import json

from utils.resume_parser import KNOWN_HARD_SKILLS, KNOWN_SOFT_SKILLS

# Паттерны для определения уровня
_SENIOR_PATTERNS  = r"\b(senior|lead|principal|staff|architect|sr\.?)\b"
_JUNIOR_PATTERNS  = r"\b(junior|entry.?level|intern|jr\.?|graduate|trainee)\b"
_MIDDLE_PATTERNS  = r"\b(middle|mid.?level|mid\b|engineer ii)\b"

# Маркеры «желательного»
_NICE_TO_HAVE = r"(nice to have|plus|preferred|advantage|bonus|желательно|будет плюсом)"
_REQUIRED     = r"(required|must|mandatory|необходимо|обязательно)"


def _detect_seniority(text: str) -> str:
    t = text.lower()
    if re.search(_SENIOR_PATTERNS, t):  return "senior"
    if re.search(_JUNIOR_PATTERNS, t):  return "junior"
    if re.search(_MIDDLE_PATTERNS, t):  return "middle"
    return "middle"  # default


def _detect_role_name(text: str) -> str:
    """Пытается извлечь название роли из первых 3 строк."""
    for line in text.strip().split("\n")[:5]:
        line = line.strip()
        if 5 < len(line) < 80 and any(w in line.lower() for w in
           ["engineer", "developer", "scientist", "analyst", "devops",
            "architect", "manager", "lead", "intern", "инженер", "разработчик"]):
            return line
    return "Unknown Role"


def _split_required_nice(text: str, skills: list[str]) -> tuple[list[str], list[str]]:
    """Разделяет навыки на обязательные и желательные по контексту в тексте."""
    required    = []
    nice_to_have = []

    lines = text.lower().split("\n")
    nice_section = False

    for line in lines:
        if re.search(_NICE_TO_HAVE, line):
            nice_section = True
        if re.search(_REQUIRED, line):
            nice_section = False

        for skill in skills:
            if skill.lower() in line:
                if nice_section:
                    if skill not in nice_to_have:
                        nice_to_have.append(skill)
                else:
                    if skill not in required:
                        required.append(skill)

    # Навыки, не попавшие ни в один раздел — обязательные
    all_found = set(required) | set(nice_to_have)
    for skill in skills:
        if skill not in all_found:
            required.append(skill)

    return required, nice_to_have


def _parse_by_regex(text: str) -> dict:
    text_lower = text.lower()
    found_skills = [s for s in KNOWN_HARD_SKILLS if s.lower() in text_lower]
    required, nice = _split_required_nice(text, found_skills)

    return {
        "role_name":       _detect_role_name(text),
        "seniority":       _detect_seniority(text),
        "required_skills": required,
        "nice_to_have":    nice,
        "_meta":           {"source": "job_parser_regex"},
    }


def _parse_by_llm(text: str) -> dict:
    if not is_available():
        raise ValueError("LLM_API_KEY не задан")

    prompt = f"""
Ты — парсер вакансий. Прочитай описание вакансии и верни ТОЛЬКО валидный JSON:
{{
  "role_name": "название роли",
  "seniority": "junior|middle|senior",
  "required_skills": ["обязательные технические навыки"],
  "nice_to_have": ["желательные навыки"],
  "responsibilities": ["ключевые обязанности (до 5 пунктов)"]
}}

Вакансия:
\"\"\"
{text[:3000]}
\"\"\"

Только JSON, без пояснений.
""".strip()

    data = call_llm_json(prompt, max_tokens=1024)

    data["_meta"] = {"source": "job_parser_llm", "provider": provider}
    return data


def parse_job_description(text: str, use_llm: bool = False) -> dict:
    """
    Парсит текст вакансии.

    Returns:
        {
          "role_name":        str,
          "seniority":        "junior"|"middle"|"senior",
          "required_skills":  list[str],
          "nice_to_have":     list[str],
          "responsibilities": list[str],   # только LLM
          "_meta":            dict,
        }
    """
    if not text or not text.strip():
        return {"role_name": "Unknown", "seniority": "middle",
                "required_skills": [], "nice_to_have": [],
                "_meta": {"source": "empty"}}

    if use_llm or os.getenv("LLM_API_KEY"):
        try:
            return _parse_by_llm(text)
        except Exception as e:
            print(f"[JobParser] LLM недоступен ({e}), использую regex.")

    return _parse_by_regex(text)


def gap_analysis(job: dict, profile: dict) -> dict:
    """
    Сравнивает требования вакансии с профилем пользователя.

    Returns:
        {
          "match_score":   float,
          "match_pct":     int,
          "you_have":      list[str],
          "you_need":      list[str],
          "nice_missing":  list[str],
          "fit_level":     "strong"|"good"|"partial"|"weak",
        }
    """
    from model.predictor import normalize_skill
    user_norm = profile.get("_all_skills_norm", set())
    if not user_norm:
        from model.predictor import normalize_skills
        user_norm = normalize_skills(
            profile.get("hard_skills", []) + profile.get("soft_skills", [])
        )

    req  = job.get("required_skills", [])
    nice = job.get("nice_to_have", [])

    you_have     = [s for s in req  if normalize_skill(s) in user_norm]
    you_need     = [s for s in req  if normalize_skill(s) not in user_norm]
    nice_missing = [s for s in nice if normalize_skill(s) not in user_norm]

    score = len(you_have) / len(req) if req else 0.0
    pct   = int(score * 100)

    if pct >= 80:   fit = "strong"
    elif pct >= 60: fit = "good"
    elif pct >= 35: fit = "partial"
    else:           fit = "weak"

    return {
        "match_score":  round(score, 4),
        "match_pct":    pct,
        "you_have":     you_have,
        "you_need":     you_need,
        "nice_missing": nice_missing,
        "fit_level":    fit,
    }


# ---------------------------------------------------------------------------
# CLI тест
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, ".")
    from model.predictor import load_profile

    sample_jd = """
    Senior Backend Engineer — Fintech Startup

    We're looking for a Backend Engineer to build our payment platform.

    Required:
    - Python (5+ years)
    - PostgreSQL, Redis
    - Docker, Kubernetes
    - REST API, Microservices
    - Git, CI/CD

    Nice to have:
    - AWS
    - Kafka
    - System Design experience

    You will: design scalable APIs, mentor junior devs, work with cloud infra.
    """

    parsed = parse_job_description(sample_jd)
    print(json.dumps({k: v for k, v in parsed.items() if not k.startswith("_")},
                     indent=2, ensure_ascii=False))

    profile = load_profile("data/profile.json")
    gap = gap_analysis(parsed, profile)
    print(f"\nMatch: {gap['match_pct']}% ({gap['fit_level']})")
    print(f"Have:  {gap['you_have']}")
    print(f"Need:  {gap['you_need']}")
