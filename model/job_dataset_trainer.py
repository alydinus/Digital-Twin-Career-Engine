"""
Model Layer -- classical ML recommender over a jobs dataset.

Supports a CSV with at least:
  - role
  - required_skills
Optional:
  - description

This gives the project a real classical-ML style path:
  user skill vector -> nearest job vectors -> top-N role predictions.
"""

from __future__ import annotations

import csv
import io
import math
import re

from model.predictor import find_missing_skills, normalize_skill, normalize_skills
from utils.job_parser import parse_job_description


def _skill_present_as_token(skill: str, text: str) -> bool:
    pattern = r"(?<![a-zA-Z0-9+#/])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9+#/])"
    return bool(re.search(pattern, text.lower()))


def load_job_dataset_csv(file_bytes: bytes) -> list[dict]:
    text = file_bytes.decode("utf-8-sig", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    jobs = []

    for row in reader:
        role = (row.get("role") or row.get("title") or "").strip()
        raw_skills = (row.get("required_skills") or row.get("skills") or "").strip()
        description = (row.get("description") or row.get("job_description") or row.get("text") or "").strip()

        if not role and not description:
            continue
        if not role:
            role = "Unknown Role"

        if raw_skills:
            required_skills = [s.strip() for s in re.split(r"[;,|/]", raw_skills) if s.strip()]
        elif description:
            parsed = parse_job_description(description, use_llm=False)
            required_skills = [
                skill
                for skill in parsed.get("required_skills", [])
                if _skill_present_as_token(skill, description)
            ]
            role = role if role != "Unknown Role" else parsed.get("role_name", role)
        else:
            required_skills = []

        jobs.append({
            "role": role,
            "required_skills": required_skills,
            "required_skills_norm": normalize_skills(required_skills),
            "description": description,
        })

    return jobs


def _cosine_similarity(a: list[int], b: list[int]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_binary_vectors(jobs: list[dict]) -> tuple[list[str], list[list[int]]]:
    vocab = sorted({skill for job in jobs for skill in job["required_skills_norm"]})
    vectors = []
    for job in jobs:
        role_skills = job["required_skills_norm"]
        vectors.append([1 if skill in role_skills else 0 for skill in vocab])
    return vocab, vectors


def predict_top_roles_knn(profile: dict, jobs: list[dict], top_n: int = 3) -> list[dict]:
    if not jobs:
        return []

    user_skills = profile.get("_all_skills_norm") or normalize_skills(
        profile.get("hard_skills", []) + profile.get("soft_skills", [])
    )
    vocab, job_vectors = _build_binary_vectors(jobs)
    user_vector = [1 if skill in user_skills else 0 for skill in vocab]

    results = []
    for job, vector in zip(jobs, job_vectors):
        score = _cosine_similarity(user_vector, vector)
        matched = [s for s in job["required_skills"] if normalize_skill(s) in user_skills]
        missing = find_missing_skills(user_skills, job["required_skills"])
        results.append({
            "role": job["role"],
            "score": round(score, 4),
            "score_pct": int(score * 100),
            "overlap_score": round(len(matched) / len(job["required_skills_norm"]), 4)
            if job["required_skills_norm"] else 0.0,
            "semantic_score": 0.0,
            "match_method": "knn_cosine",
            "semantic_matches": [],
            "matched_skills": matched,
            "missing_skills": missing,
            "description": job.get("description", ""),
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_n]
