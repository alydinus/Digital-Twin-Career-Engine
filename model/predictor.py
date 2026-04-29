"""
Model Layer — предсказание карьерных направлений на основе навыков.

Алгоритм:
  match_score = |user_skills cap role_skills| / |role_skills|

Future (hook): TF-IDF или embedding-матчинг для похожих навыков.
"""

from __future__ import annotations

import json
import csv
import re
from pathlib import Path

from model.semantic_matcher import SKLEARN_AVAILABLE, hybrid_score, semantic_match_score

# ---------------------------------------------------------------------------
# Нормализация навыков
# ---------------------------------------------------------------------------

SKILL_ALIASES = {
    "kubernetes": ["k8s"],
    "pytorch":    ["torch"],
    "postgresql": ["postgres", "psql"],
    "javascript": ["js"],
    "typescript": ["ts"],
    "machine learning": ["ml"],
    "python":     ["py"],
    "ci/cd":      ["cicd", "ci-cd", "gitlab ci", "github actions"],
    "rest api":   ["rest", "restful api", "restful"],
    "scikit-learn": ["sklearn"],
    "matplotlib": ["plt", "mpl"],
}

_ALIAS_MAP = {}
for canonical, aliases in SKILL_ALIASES.items():
    for alias in aliases:
        _ALIAS_MAP[alias] = canonical


def normalize_skill(skill):
    s = skill.strip().lower()
    return _ALIAS_MAP.get(s, s)


def normalize_skills(skills):
    return {normalize_skill(s) for s in skills if s.strip()}


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_profile(path="data/profile.json"):
    """
    Загружает профиль пользователя из JSON.
    Future hook: подключить парсер резюме — вернуть dict с теми же ключами.
    """
    with open(path, encoding="utf-8") as f:
        profile = json.load(f)
    profile["_hard_skills_norm"] = normalize_skills(profile.get("hard_skills", []))
    profile["_soft_skills_norm"] = normalize_skills(profile.get("soft_skills", []))
    profile["_all_skills_norm"]  = (
        profile["_hard_skills_norm"] | profile["_soft_skills_norm"]
    )
    return profile


def load_jobs(path="data/jobs.csv"):
    """
    Загружает базу профессий из CSV.
    Future hook: заменить на запрос к базе/API.
    """
    jobs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_skills = re.split(r"[;,]", row.get("required_skills", ""))
            raw_skills = [s.strip() for s in raw_skills if s.strip()]
            jobs.append({
                "role":                 row["role"].strip(),
                "required_skills":      raw_skills,
                "required_skills_norm": normalize_skills(raw_skills),
                "description":          row.get("description", "").strip(),
            })
    return jobs


# ---------------------------------------------------------------------------
# ML-матчинг
# ---------------------------------------------------------------------------

def compute_match_score(user_skills_norm, role_skills_norm):
    """Overlap score = |user cap role| / |role|. Returns 0.0..1.0."""
    if not role_skills_norm:
        return 0.0
    matched = user_skills_norm & role_skills_norm
    return round(len(matched) / len(role_skills_norm), 4)


def find_missing_skills(user_skills_norm, role_skills, role_skills_norm=None):
    """
    Возвращает недостающие навыки в оригинальном регистре.
    Нормализует каждый навык индивидуально (set не сохраняет порядок).
    """
    return [s for s in role_skills if normalize_skill(s) not in user_skills_norm]


def predict_top_roles(profile, jobs, top_n=3, use_semantic=True):
    """
    Возвращает топ-N профессий со score и missing skills.

    Если доступен scikit-learn, комбинирует:
      - overlap score   (точные совпадения)
      - semantic score  (TF-IDF similarity)
    в итоговый hybrid score.
    """
    user_skills = profile["_all_skills_norm"]
    user_skill_list = profile.get("hard_skills", []) + profile.get("soft_skills", [])
    results = []
    for job in jobs:
        role_norm = job["required_skills_norm"]
        overlap = compute_match_score(user_skills, role_norm)
        semantic = 0.0
        semantic_matches = []
        match_method = "overlap"

        if use_semantic and SKLEARN_AVAILABLE and user_skill_list:
            semantic_result = semantic_match_score(user_skill_list, job["required_skills"])
            semantic = semantic_result["score"]
            semantic_matches = semantic_result["semantic_matches"]
            score = hybrid_score(overlap, semantic)
            match_method = "hybrid_tfidf"
        else:
            score = overlap

        matched = [s for s in job["required_skills"] if normalize_skill(s) in user_skills]
        missing = find_missing_skills(user_skills, job["required_skills"])
        results.append({
            "role":           job["role"],
            "score":          score,
            "score_pct":      int(score * 100),
            "overlap_score":  overlap,
            "semantic_score": semantic,
            "match_method":   match_method,
            "semantic_matches": semantic_matches,
            "matched_skills": matched,
            "missing_skills": missing,
            "description":    job["description"],
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    base    = Path(__file__).parent.parent
    profile = load_profile(base / "data" / "profile.json")
    jobs    = load_jobs(base / "data" / "jobs.csv")
    top     = predict_top_roles(profile, jobs)

    print(f"\nПрофиль: {profile['name']}")
    print(f"Hard skills: {', '.join(profile['hard_skills'])}\n")
    print("Топ профессий:\n")
    for i, r in enumerate(top, 1):
        bar = "#" * r["score_pct"] + "." * (100 - r["score_pct"])
        print(f"  {i}. {r['role']:25s}  {r['score_pct']:3d}%  |{bar[:30]}|")
        print(f"     OK:   {', '.join(r['matched_skills']) or 'нет'}")
        print(f"     Нет:  {', '.join(r['missing_skills']) or 'нет'}")
        print()
