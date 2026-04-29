import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.job_dataset_trainer import load_job_dataset_csv, predict_top_roles_knn
from utils.profile_ingestion import (
    merge_profiles,
    parse_academic_transcript,
    parse_linkedin_profile,
)


def _profile(skills):
    from model.predictor import normalize_skills

    ns = normalize_skills(skills)
    return {
        "name": "Test User",
        "hard_skills": skills,
        "soft_skills": [],
        "interests": [],
        "_hard_skills_norm": ns,
        "_soft_skills_norm": set(),
        "_all_skills_norm": ns,
    }


def test_parse_linkedin_profile_extracts_skills():
    text = "Backend Engineer. Skills: Python, Docker, PostgreSQL, Git. Interests: backend, cloud"
    parsed = parse_linkedin_profile(text, use_llm=False)
    assert "Python" in parsed["hard_skills"]
    assert parsed["_meta"]["source"] == "linkedin_parser"


def test_parse_transcript_extracts_skills_and_gpa():
    text = "Courses: Machine Learning, Statistics, Database Systems, Cloud Computing. GPA: 3.8/4.0"
    parsed = parse_academic_transcript(text)
    assert "Machine Learning" in parsed["hard_skills"]
    assert "SQL" in parsed["hard_skills"]
    assert parsed["_meta"]["gpa"] == 3.8


def test_merge_profiles_unions_sources():
    p1 = {"name": "Alydin", "hard_skills": ["Python"], "soft_skills": [], "interests": [], "_meta": {"source": "resume"}}
    p2 = {"name": "User", "hard_skills": ["Docker"], "soft_skills": [], "interests": ["cloud"], "_meta": {"source": "linkedin"}}
    merged = merge_profiles([p1, p2])
    assert merged["name"] == "Alydin"
    assert "Python" in merged["hard_skills"]
    assert "Docker" in merged["hard_skills"]
    assert "resume" in merged["_meta"]["sources"]
    assert "linkedin" in merged["_meta"]["sources"]


def test_load_job_dataset_csv_and_knn_prediction():
    csv_bytes = (
        b"role,required_skills,description\n"
        b"Backend Engineer,\"Python;Docker;PostgreSQL;Git\",Backend role\n"
        b"DevOps Engineer,\"Docker;Kubernetes;AWS;Linux\",DevOps role\n"
    )
    jobs = load_job_dataset_csv(csv_bytes)
    results = predict_top_roles_knn(_profile(["Python", "Docker", "Git"]), jobs, top_n=2)
    assert len(jobs) == 2
    assert results[0]["role"] == "Backend Engineer"
    assert results[0]["match_method"] == "knn_cosine"
