"""
Agent Layer — Career Coach (GenAI)

Генерирует персонализированный план развития карьеры на основе:
  - текущих навыков пользователя
  - целевой роли
  - недостающих навыков

Режимы:
  1. Template fallback — структурированный шаблон с приоритетами
  2. LLM mode         — генеративный персональный roadmap через Claude/GPT
"""

from __future__ import annotations
from utils.llm_client import call_llm_json, is_available
import os
from typing import Optional

# ---------------------------------------------------------------------------
# Шаблонный генератор (fallback)
# ---------------------------------------------------------------------------

# Сложность навыка (1=просто, 3=сложно)
SKILL_DIFFICULTY = {
    "git": 1, "linux": 1, "bash": 1, "html": 1, "css": 1,
    "python": 1, "javascript": 1, "sql": 1, "docker": 2,
    "rest api": 1, "react": 2, "postgresql": 2, "typescript": 2,
    "system design": 3, "microservices": 3, "kubernetes": 3,
    "aws": 2, "terraform": 2, "ci/cd": 2, "kafka": 3,
    "machine learning": 3, "pytorch": 3, "mlops": 3,
    "statistics": 2, "math": 2, "numpy": 1, "pandas": 1,
    "scikit-learn": 2, "figma": 1, "agile": 1,
}

# Сколько недель на изучение (примерно)
SKILL_WEEKS = {
    "git": 1, "linux": 2, "bash": 1, "html": 1, "css": 2,
    "python": 4, "javascript": 6, "sql": 3, "docker": 2,
    "rest api": 1, "react": 6, "postgresql": 2, "typescript": 3,
    "system design": 8, "microservices": 6, "kubernetes": 4,
    "aws": 6, "terraform": 3, "ci/cd": 2, "kafka": 3,
    "machine learning": 10, "pytorch": 8, "mlops": 5,
    "statistics": 6, "math": 8, "numpy": 1, "pandas": 2,
    "scikit-learn": 3, "figma": 2, "agile": 1,
}

DEFAULT_WEEKS = 3


def _generate_template_roadmap(
    profile: dict,
    target_role: str,
    missing_skills: list[str],
    matched_skills: list[str],
    score_pct: int,
) -> dict:
    """Генерирует структурированный roadmap без LLM."""

    # Сортируем: сначала простые (быстрые победы), потом сложные
    def sort_key(skill):
        k = skill.lower()
        return (SKILL_DIFFICULTY.get(k, 2), SKILL_WEEKS.get(k, DEFAULT_WEEKS))

    sorted_missing = sorted(missing_skills, key=sort_key)

    # Строим этапы по 4-6 недель
    phases = []
    current_phase = {"title": "", "skills": [], "weeks": 0}
    phase_num = 1

    for skill in sorted_missing:
        weeks = SKILL_WEEKS.get(skill.lower(), DEFAULT_WEEKS)
        if current_phase["weeks"] + weeks > 6 and current_phase["skills"]:
            current_phase["title"] = f"Фаза {phase_num}: {', '.join(current_phase['skills'][:2])}..."
            phases.append(current_phase)
            phase_num += 1
            current_phase = {"title": "", "skills": [], "weeks": 0}
        current_phase["skills"].append(skill)
        current_phase["weeks"] += weeks

    if current_phase["skills"]:
        current_phase["title"] = f"Фаза {phase_num}: {', '.join(current_phase['skills'][:2])}..."
        phases.append(current_phase)

    total_weeks = sum(p["weeks"] for p in phases)
    months      = round(total_weeks / 4, 1)

    # Советы на основе текущего score
    if score_pct >= 70:
        readiness = "Ты уже хорошо подходишь для этой роли. Фокусируйся на углублении."
        priority  = "high"
    elif score_pct >= 40:
        readiness = "Есть хорошая база. Несколько целевых навыков выведут тебя на уровень."
        priority  = "medium"
    else:
        readiness = "Роль амбициозная. Рекомендую начать с основ и двигаться поэтапно."
        priority  = "low"

    return {
        "target_role":     target_role,
        "current_score":   score_pct,
        "readiness":       readiness,
        "readiness_level": priority,
        "total_weeks":     total_weeks,
        "total_months":    months,
        "phases":          phases,
        "quick_wins":      [s for s in sorted_missing if SKILL_DIFFICULTY.get(s.lower(), 2) == 1][:3],
        "strengths":       matched_skills,
        "source":          "template",
    }


# ---------------------------------------------------------------------------
# LLM Career Coach
# ---------------------------------------------------------------------------

def _generate_llm_roadmap(
    profile: dict,
    target_role: str,
    missing_skills: list[str],
    matched_skills: list[str],
    score_pct: int,
) -> dict:
    """Генерирует персонализированный roadmap через LLM."""

    if not is_available():
        raise ValueError("LLM_API_KEY не задан")

    prompt = f"""
Ты — опытный карьерный коуч в IT. Составь персонализированный план развития.

Профиль:
- Имя: {profile.get('name', 'пользователь')}
- Текущие навыки: {', '.join(matched_skills)}
- Целевая роль: {target_role}
- Текущий match score: {score_pct}%
- Недостающие навыки: {', '.join(missing_skills)}

Верни ТОЛЬКО валидный JSON:
{{
  "target_role": "{target_role}",
  "current_score": {score_pct},
  "readiness": "краткая оценка готовности (1-2 предложения)",
  "readiness_level": "high|medium|low",
  "total_weeks": число_недель,
  "total_months": число_месяцев_float,
  "phases": [
    {{
      "title": "Фаза 1: Название",
      "skills": ["навык1", "навык2"],
      "weeks": N,
      "description": "что делать в этой фазе"
    }}
  ],
  "quick_wins": ["самые быстрые навыки для изучения"],
  "strengths": {json_strengths},
  "tips": ["персональный совет 1", "персональный совет 2"],
  "source": "llm"
}}
""".strip().replace("{json_strengths}", str(matched_skills))

    return call_llm_json(prompt, max_tokens=1500)


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def generate_roadmap(
    profile: dict,
    target_role: str,
    missing_skills: list[str],
    matched_skills: list[str],
    score_pct: int,
    use_llm: bool = False,
) -> dict:
    """
    Генерирует персонализированный карьерный roadmap.

    Args:
        profile:        профиль пользователя
        target_role:    целевая профессия
        missing_skills: список навыков для изучения
        matched_skills: текущие совпавшие навыки
        score_pct:      текущий match score (0-100)
        use_llm:        форсировать LLM

    Returns:
        dict с полями: target_role, phases, quick_wins, total_months, source, ...
    """
    if not missing_skills:
        return {
            "target_role":     target_role,
            "current_score":   score_pct,
            "readiness":       "Все навыки уже есть! Ты готов к этой роли.",
            "readiness_level": "high",
            "total_weeks":     0,
            "total_months":    0,
            "phases":          [],
            "quick_wins":      [],
            "strengths":       matched_skills,
            "source":          "template",
        }

    # Пробуем LLM
    if use_llm or os.getenv("LLM_API_KEY"):
        try:
            return _generate_llm_roadmap(
                profile, target_role, missing_skills, matched_skills, score_pct
            )
        except Exception as e:
            print(f"[CareerCoach] LLM недоступен ({e}), использую шаблон.")

    return _generate_template_roadmap(
        profile, target_role, missing_skills, matched_skills, score_pct
    )


# ---------------------------------------------------------------------------
# CLI тест
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, ".")
    from model.predictor import load_profile, load_jobs, predict_top_roles

    profile = load_profile("data/profile.json")
    jobs    = load_jobs("data/jobs.csv")
    top     = predict_top_roles(profile, jobs, top_n=1)
    best    = top[0]

    roadmap = generate_roadmap(
        profile        = profile,
        target_role    = best["role"],
        missing_skills = best["missing_skills"],
        matched_skills = best["matched_skills"],
        score_pct      = best["score_pct"],
    )

    print(json.dumps(roadmap, indent=2, ensure_ascii=False))
