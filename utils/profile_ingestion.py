"""
Platform Layer -- extra source ingestion for LinkedIn and academic transcript.

PDF требует не только CV, но и LinkedIn profile + Academic Transcript.
Этот модуль превращает эти источники в тот же profile schema, что и resume_parser.
"""

from __future__ import annotations

import re

from utils.resume_parser import parse_resume


_TRANSCRIPT_RULES = {
    "data structures": {
        "skills": ["Algorithms"],
        "interests": ["backend"],
    },
    "algorithms": {
        "skills": ["Algorithms"],
        "interests": ["backend"],
    },
    "database": {
        "skills": ["SQL", "PostgreSQL"],
        "interests": ["data", "backend"],
    },
    "operating systems": {
        "skills": ["Linux"],
        "interests": ["backend"],
    },
    "computer networks": {
        "skills": ["Linux"],
        "interests": ["security"],
    },
    "software engineering": {
        "skills": ["Git", "System Design"],
        "interests": ["backend"],
    },
    "web development": {
        "skills": ["HTML", "CSS", "JavaScript"],
        "interests": ["frontend"],
    },
    "machine learning": {
        "skills": ["Machine Learning", "Scikit-learn"],
        "interests": ["AI", "ML", "data"],
    },
    "deep learning": {
        "skills": ["PyTorch", "TensorFlow"],
        "interests": ["AI", "ML"],
    },
    "artificial intelligence": {
        "skills": ["Machine Learning"],
        "interests": ["AI", "ML"],
    },
    "data mining": {
        "skills": ["Pandas", "SQL"],
        "interests": ["data"],
    },
    "statistics": {
        "skills": ["Statistics", "Math"],
        "interests": ["data", "ML"],
    },
    "probability": {
        "skills": ["Statistics", "Math"],
        "interests": ["data", "ML"],
    },
    "linear algebra": {
        "skills": ["Math"],
        "interests": ["ML"],
    },
    "calculus": {
        "skills": ["Math"],
        "interests": ["ML"],
    },
    "distributed systems": {
        "skills": ["Microservices", "Kafka"],
        "interests": ["backend", "distributed systems"],
    },
    "cloud computing": {
        "skills": ["AWS", "Docker"],
        "interests": ["cloud", "backend"],
    },
    "information security": {
        "skills": ["Linux"],
        "interests": ["security"],
    },
}


def _unique(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        item = (item or "").strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def parse_linkedin_profile(text: str, use_llm: bool = False) -> dict:
    profile = parse_resume(text, use_llm=use_llm)
    name = (profile.get("name") or "").strip()
    if ":" in name or len(name.split()) > 5:
        profile["name"] = "User"
    profile["_meta"]["source"] = "linkedin_parser"
    return profile


def parse_academic_transcript(text: str) -> dict:
    lower = (text or "").lower()
    skills: list[str] = []
    interests: list[str] = []
    courses: list[str] = []

    for course_name, payload in _TRANSCRIPT_RULES.items():
        if course_name in lower:
            courses.append(course_name.title())
            skills.extend(payload["skills"])
            interests.extend(payload["interests"])

    gpa = None
    gpa_match = re.search(r"\b(?:gpa|cgpa|grade point average)\s*[:=]?\s*([0-4](?:\.\d{1,2})?)", lower)
    if gpa_match:
        gpa = float(gpa_match.group(1))
    else:
        slash_match = re.search(r"\b([0-4](?:\.\d{1,2})?)\s*/\s*4(?:\.0)?\b", lower)
        if slash_match:
            gpa = float(slash_match.group(1))

    semester_count = len(re.findall(r"\bsemester\b|\bterm\b|\bсеместр\b", lower))

    return {
        "name": "User",
        "hard_skills": _unique(skills),
        "soft_skills": [],
        "interests": _unique(interests),
        "_meta": {
            "source": "transcript_parser",
            "gpa": gpa,
            "courses_detected": _unique(courses),
            "semester_mentions": semester_count,
        },
    }


def merge_profiles(profiles: list[dict]) -> dict:
    name = "User"
    hard: list[str] = []
    soft: list[str] = []
    interests: list[str] = []
    sources: list[str] = []
    transcript_meta = {}

    for profile in profiles:
        if not profile:
            continue
        current_name = (profile.get("name") or "").strip()
        if current_name and current_name.lower() != "user" and name == "User":
            name = current_name
        hard.extend(profile.get("hard_skills", []))
        soft.extend(profile.get("soft_skills", []))
        interests.extend(profile.get("interests", []))

        meta = profile.get("_meta", {})
        if meta.get("source"):
            sources.append(meta["source"])
        if meta.get("gpa") is not None:
            transcript_meta["gpa"] = meta["gpa"]
        if meta.get("courses_detected"):
            transcript_meta["courses_detected"] = meta["courses_detected"]

    return {
        "name": name,
        "hard_skills": _unique(hard),
        "soft_skills": _unique(soft),
        "interests": _unique(interests),
        "_meta": {
            "source": "merged_profile",
            "sources": _unique(sources),
            **transcript_meta,
        },
    }
