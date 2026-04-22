"""
Unit-тесты для Model Layer.

Запуск: pytest tests/ -v
"""

import sys
from pathlib import Path

# Добавляем корень проекта в path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.predictor import (
    normalize_skill,
    normalize_skills,
    compute_match_score,
    find_missing_skills,
    predict_top_roles,
    load_profile,
    load_jobs,
)


# ---------------------------------------------------------------------------
# Нормализация
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_lowercase(self):
        assert normalize_skill("Python") == "python"

    def test_alias_k8s(self):
        assert normalize_skill("k8s") == "kubernetes"

    def test_alias_torch(self):
        assert normalize_skill("torch") == "pytorch"

    def test_alias_ml(self):
        assert normalize_skill("ML") == "machine learning"

    def test_unknown_skill(self):
        assert normalize_skill("SomeNewSkill") == "somewskill".replace("w", "ne")

    def test_normalize_skills_set(self):
        result = normalize_skills(["Python", "Docker", "k8s"])
        assert "python" in result
        assert "docker" in result
        assert "kubernetes" in result


# ---------------------------------------------------------------------------
# Score и missing skills
# ---------------------------------------------------------------------------

class TestMatchScore:
    def test_perfect_match(self):
        user  = {"python", "docker", "sql"}
        role  = {"python", "docker", "sql"}
        assert compute_match_score(user, role) == 1.0

    def test_zero_match(self):
        user  = {"python", "docker"}
        role  = {"javascript", "react"}
        assert compute_match_score(user, role) == 0.0

    def test_partial_match(self):
        user  = {"python", "docker"}
        role  = {"python", "docker", "kubernetes", "aws"}
        score = compute_match_score(user, role)
        assert score == 0.5  # 2/4

    def test_empty_role(self):
        assert compute_match_score({"python"}, set()) == 0.0

    def test_rounding(self):
        user  = {"a"}
        role  = {"a", "b", "c"}
        # 1/3 = 0.3333...  -> rounded to 4 decimal = 0.3333
        assert compute_match_score(user, role) == round(1/3, 4)


class TestMissingSkills:
    def test_all_missing(self):
        user   = set()
        skills = ["Docker", "Kubernetes"]
        skills_norm = {"docker", "kubernetes"}
        missing = find_missing_skills(user, skills, skills_norm)
        assert missing == ["Docker", "Kubernetes"]

    def test_none_missing(self):
        user   = {"docker", "kubernetes"}
        skills = ["Docker", "Kubernetes"]
        skills_norm = {"docker", "kubernetes"}
        missing = find_missing_skills(user, skills, skills_norm)
        assert missing == []

    def test_partial_missing(self):
        user   = {"python"}
        skills = ["Python", "Docker", "Kubernetes"]
        skills_norm = {"python", "docker", "kubernetes"}
        missing = find_missing_skills(user, skills, skills_norm)
        assert "Python" not in missing
        assert "Docker" in missing
        assert "Kubernetes" in missing


# ---------------------------------------------------------------------------
# Predict top roles (интеграционный тест на фиктивном профиле)
# ---------------------------------------------------------------------------

class TestPredictTopRoles:
    def _make_profile(self, hard_skills):
        from model.predictor import normalize_skills
        ns = normalize_skills(hard_skills)
        return {
            "name": "Test User",
            "hard_skills": hard_skills,
            "soft_skills": [],
            "interests": [],
            "_hard_skills_norm": ns,
            "_soft_skills_norm": set(),
            "_all_skills_norm": ns,
        }

    def _make_jobs(self):
        return [
            {
                "role": "Backend Engineer",
                "required_skills": ["Python", "SQL", "Docker", "Git"],
                "required_skills_norm": {"python", "sql", "docker", "git"},
                "description": "Backend dev",
            },
            {
                "role": "DevOps Engineer",
                "required_skills": ["Docker", "Kubernetes", "AWS"],
                "required_skills_norm": {"docker", "kubernetes", "aws"},
                "description": "DevOps",
            },
            {
                "role": "Frontend Engineer",
                "required_skills": ["JavaScript", "React", "CSS"],
                "required_skills_norm": {"javascript", "react", "css"},
                "description": "Frontend dev",
            },
        ]

    def test_top_n_limit(self):
        profile = self._make_profile(["Python", "SQL", "Docker", "Git"])
        jobs    = self._make_jobs()
        results = predict_top_roles(profile, jobs, top_n=2)
        assert len(results) == 2

    def test_best_match_first(self):
        profile = self._make_profile(["Python", "SQL", "Docker", "Git"])
        jobs    = self._make_jobs()
        results = predict_top_roles(profile, jobs, top_n=3)
        # Backend Engineer должен быть на первом месте (4/4 = 1.0)
        assert results[0]["role"] == "Backend Engineer"
        assert results[0]["score"] == 1.0

    def test_scores_sorted_desc(self):
        profile = self._make_profile(["Python", "Docker"])
        jobs    = self._make_jobs()
        results = predict_top_roles(profile, jobs, top_n=3)
        scores  = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_missing_skills_correct(self):
        profile = self._make_profile(["Python"])
        jobs    = self._make_jobs()
        results = predict_top_roles(profile, jobs, top_n=3)
        backend = next(r for r in results if r["role"] == "Backend Engineer")
        assert "SQL" in backend["missing_skills"]
        assert "Python" not in backend["missing_skills"]

    def test_zero_score_role(self):
        profile = self._make_profile(["Python"])
        jobs    = self._make_jobs()
        results = predict_top_roles(profile, jobs, top_n=3)
        frontend = next(r for r in results if r["role"] == "Frontend Engineer")
        assert frontend["score"] == 0.0
        assert set(frontend["missing_skills"]) == {"JavaScript", "React", "CSS"}


# ---------------------------------------------------------------------------
# Интеграция с реальными файлами данных
# ---------------------------------------------------------------------------

class TestRealData:
    def test_load_profile(self):
        profile = load_profile(ROOT / "data" / "profile.json")
        assert "name" in profile
        assert isinstance(profile["hard_skills"], list)
        assert "_all_skills_norm" in profile

    def test_load_jobs(self):
        jobs = load_jobs(ROOT / "data" / "jobs.csv")
        assert len(jobs) >= 5
        for job in jobs:
            assert "role" in job
            assert "required_skills_norm" in job

    def test_full_pipeline(self):
        profile = load_profile(ROOT / "data" / "profile.json")
        jobs    = load_jobs(ROOT / "data" / "jobs.csv")
        results = predict_top_roles(profile, jobs, top_n=3)
        assert len(results) == 3
        for r in results:
            assert 0.0 <= r["score"] <= 1.0
            assert isinstance(r["missing_skills"], list)
            assert isinstance(r["matched_skills"], list)
