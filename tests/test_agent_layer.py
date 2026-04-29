import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agent.resource_finder import _build_live_query, _categorize_result, find_resources
from model.predictor import predict_top_roles
from utils.notebooklm_bridge import (
    build_antigravity_system_prompt,
    build_notebooklm_bridge_config,
    validate_notebooklm_url,
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


def _jobs():
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
    ]


def test_predictor_returns_hybrid_metadata():
    results = predict_top_roles(_profile(["Python", "Docker", "Git"]), _jobs(), top_n=2)
    top = results[0]
    assert "overlap_score" in top
    assert "semantic_score" in top
    assert "match_method" in top
    assert 0.0 <= top["score"] <= 1.0


def test_live_query_includes_role_context():
    query = _build_live_query("Kubernetes", "DevOps Engineer")
    assert "Kubernetes" in query
    assert "DevOps Engineer" in query


def test_categorize_result_detects_github():
    category = _categorize_result(
        "https://github.com/kelseyhightower/kubernetes-the-hard-way",
        "GitHub - kubernetes the hard way",
        "Hands-on repo",
    )
    assert category == "github"


def test_find_resources_static_fallback():
    result = find_resources("Docker")
    assert result["source"] == "static"
    assert any("docker" in url.lower() for url in result["docs"])


def test_validate_notebooklm_url():
    assert validate_notebooklm_url("https://notebooklm.google.com/notebook/abc123")
    assert not validate_notebooklm_url("https://example.com/notebook/abc123")


def test_build_notebooklm_bridge_config():
    notebook_url = "https://notebooklm.google.com/notebook/abc123"
    config = build_notebooklm_bridge_config(
        notebook_url,
        target_role="ML Engineer",
        missing_skill="MLOps",
    )
    prompt = build_antigravity_system_prompt(
        notebook_url,
        target_role="ML Engineer",
        missing_skill="MLOps",
    )
    assert config["notebook_url"] == notebook_url
    assert config["mcp_server_package"] == "notebooklm-mcp-server"
    assert "ML Engineer" in prompt
    assert "MLOps" in prompt
