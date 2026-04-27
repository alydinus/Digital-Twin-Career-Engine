"""
Model Layer — Semantic Skill Matcher

Дополнение к точному overlap-матчингу: считает схожесть навыков через
TF-IDF + cosine similarity. Позволяет находить «почти совпадения»:
  "Machine Learning" ~ "ML Engineer"
  "PyTorch"          ~ "Deep Learning"

Режимы:
  1. TF-IDF cosine (scikit-learn) — основной
  2. Fallback                     — обычный overlap если sklearn недоступен
"""

from __future__ import annotations
from typing import Optional

# ---------------------------------------------------------------------------
# Попытка импорта sklearn
# ---------------------------------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def semantic_match_score(
    user_skills: list[str],
    role_skills: list[str],
    threshold: float = 0.35,
) -> dict:
    """
    Вычисляет семантическую схожесть навыков пользователя с требованиями роли.

    Returns:
        {
          "score":           float,   # взвешенный semantic score 0..1
          "semantic_matches": list,   # [(user_skill, role_skill, sim_score)]
          "unmatched_role":  list,    # навыки роли без семантического аналога
          "method":          str,     # "tfidf" | "overlap_fallback"
        }
    """
    if not user_skills or not role_skills:
        return {"score": 0.0, "semantic_matches": [], "unmatched_role": role_skills, "method": "empty"}

    if not SKLEARN_AVAILABLE:
        # Fallback: обычное пересечение строк
        u_set = {s.lower() for s in user_skills}
        r_set = {s.lower() for s in role_skills}
        matched = u_set & r_set
        score = len(matched) / len(r_set) if r_set else 0.0
        return {
            "score":            round(score, 4),
            "semantic_matches": [(s, s, 1.0) for s in matched],
            "unmatched_role":   [s for s in role_skills if s.lower() not in u_set],
            "method":           "overlap_fallback",
        }

    # TF-IDF vectorization по отдельным навыкам
    all_skills = user_skills + role_skills
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    try:
        tfidf_matrix = vectorizer.fit_transform(all_skills)
    except Exception:
        return {"score": 0.0, "semantic_matches": [], "unmatched_role": role_skills, "method": "error"}

    n_user = len(user_skills)
    user_vecs = tfidf_matrix[:n_user]
    role_vecs = tfidf_matrix[n_user:]

    # Матрица схожести [n_user x n_role]
    sim_matrix = cosine_similarity(user_vecs, role_vecs)

    semantic_matches = []
    unmatched_role   = []

    for j, r_skill in enumerate(role_skills):
        best_idx = int(np.argmax(sim_matrix[:, j]))
        best_sim = float(sim_matrix[best_idx, j])
        if best_sim >= threshold:
            semantic_matches.append((user_skills[best_idx], r_skill, round(best_sim, 3)))
        else:
            unmatched_role.append(r_skill)

    score = len(semantic_matches) / len(role_skills) if role_skills else 0.0

    return {
        "score":            round(score, 4),
        "semantic_matches": semantic_matches,
        "unmatched_role":   unmatched_role,
        "method":           "tfidf",
    }


def hybrid_score(
    overlap_score: float,
    semantic_score: float,
    alpha: float = 0.6,
) -> float:
    """
    Взвешенная комбинация точного и семантического score.
    alpha=0.6 => 60% точный матчинг + 40% семантический
    """
    return round(alpha * overlap_score + (1 - alpha) * semantic_score, 4)


# ---------------------------------------------------------------------------
# CLI тест
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    user  = ["Python", "ML", "Docker", "Data Analysis"]
    role  = ["Python", "Machine Learning", "Docker", "Statistics", "Deep Learning"]

    result = semantic_match_score(user, role)
    print(f"Method: {result['method']}")
    print(f"Semantic score: {result['score']}")
    print("Matches:")
    for u, r, sim in result["semantic_matches"]:
        print(f"  '{u}' ~ '{r}'  ({sim:.3f})")
    print(f"Unmatched role skills: {result['unmatched_role']}")
