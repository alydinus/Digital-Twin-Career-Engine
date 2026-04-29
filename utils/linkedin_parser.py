"""
Utils — LinkedIn Profile Parser

Two input modes:
  1. LinkedIn PDF export (Settings -> Data Privacy -> Get a copy)
  2. Raw text (copy-paste from LinkedIn profile page)

Output format matches resume_parser.py profile dict.
"""

from __future__ import annotations
import os, re, json
from typing import Optional

_HARD_SKILLS = {
    "python","java","javascript","typescript","go","golang","rust","c++","c#",
    "kotlin","swift","scala","php","ruby","r","sql","bash","shell",
    "react","vue","angular","next.js","svelte","django","flask","fastapi",
    "spring","spring boot","express","node.js",".net","flutter","react native",
    "pandas","numpy","scikit-learn","pytorch","tensorflow","keras","matplotlib",
    "jupyter","spark","airflow","machine learning","deep learning","nlp",
    "computer vision","statistics","math","data analysis","mlops","llm",
    "postgresql","mysql","mongodb","redis","sqlite","elasticsearch","clickhouse",
    "cassandra","dynamodb","oracle","neo4j",
    "docker","kubernetes","k8s","aws","gcp","azure","terraform","ansible",
    "helm","jenkins","ci/cd","github actions","gitlab ci","prometheus",
    "grafana","nginx","kafka","rabbitmq","linux","unix",
    "git","github","gitlab","jira","confluence","figma","postman","swagger",
    "rest api","graphql","grpc","microservices","system design","agile",
    "scrum","devops","blockchain","cybersecurity",
}

_SOFT_SKILLS = {
    "communication","teamwork","leadership","problem solving",
    "critical thinking","creativity","adaptability","time management",
    "collaboration","mentoring","coaching","public speaking",
    "negotiation","empathy","ownership","persistence","initiative",
    "strategic thinking","project management",
    "коммуникация","лидерство","командная работа",
}


def _extract_name_from_text(text: str) -> str:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        first = lines[0]
        skip = {"linkedin","profile","experience","education","skills","summary","about","contact"}
        if first.lower() not in skip and len(first.split()) <= 5:
            return first
    return ""


def _extract_skills_regex(text: str):
    text_lower = text.lower()
    hard, soft = [], []
    fixes = {"Postgresql":"PostgreSQL","Mongodb":"MongoDB","Mysql":"MySQL",
             "Graphql":"GraphQL","Rest Api":"REST API","Ci/Cd":"CI/CD",
             "Aws":"AWS","Gcp":"GCP","Mlops":"MLOps","Devops":"DevOps",
             "Github":"GitHub","Gitlab":"GitLab","Grpc":"gRPC","Nlp":"NLP",
             "Llm":"LLM","Html":"HTML","Css":"CSS","Node.Js":"Node.js",
             "Next.Js":"Next.js","Fastapi":"FastAPI","Pytorch":"PyTorch",
             "Tensorflow":"TensorFlow","Numpy":"NumPy","Scikit-Learn":"Scikit-learn",
             "Neo4J":"Neo4j"}
    for skill in sorted(_HARD_SKILLS, key=len, reverse=True):
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            display = skill.title() if len(skill) > 3 else skill.upper()
            display = fixes.get(display, display)
            if display not in hard:
                hard.append(display)
    for skill in _SOFT_SKILLS:
        if skill in text_lower:
            display = skill.title()
            if display not in soft:
                soft.append(display)
    return hard, soft


def _extract_sections(text: str) -> dict:
    sections = {"experience": [], "education": [], "certifications": []}
    headers = {
        "experience": r"(?i)\b(experience|опыт работы|опыт)\b",
        "education": r"(?i)\b(education|образование)\b",
        "certifications": r"(?i)\b(certifications?|licenses?|сертификат)",
    }
    for key, pattern in headers.items():
        match = re.search(pattern, text)
        if match:
            start = match.end()
            remaining = text[start:]
            nxt = re.search(
                r"(?i)\n\s*(experience|education|skills|certifications?|"
                r"projects?|volunteer|honors|languages|interests)\s*\n",
                remaining)
            end = nxt.start() if nxt else len(remaining)
            section_text = remaining[:end].strip()
            items = re.split(r"\n\s*\n|\n\s*[•\-\*]\s*", section_text)
            sections[key] = [i.strip() for i in items if i.strip()]
    return sections


def _extract_interests(text: str) -> list[str]:
    imap = {
        "backend": ["backend","серверная","api","microservice"],
        "frontend": ["frontend","react","vue","angular","ui/ux"],
        "data science": ["data science","machine learning","ml","анализ данных"],
        "AI": ["artificial intelligence","ai","deep learning","neural"],
        "devops": ["devops","infrastructure","cloud","kubernetes"],
        "security": ["security","cybersecurity","безопасность"],
        "mobile": ["mobile","ios","android","flutter"],
    }
    text_lower = text.lower()
    return [k for k, kws in imap.items() if any(w in text_lower for w in kws)]


def _parse_linkedin_llm(text: str) -> dict:
    from utils.llm_client import call_llm_json
    prompt = (
        "Parse this LinkedIn profile. Extract ALL skills.\n\n"
        f"TEXT:\n{text[:4000]}\n\n"
        "Return JSON: {\"name\": str, \"hard_skills\": [...], "
        "\"soft_skills\": [...], \"interests\": [...], "
        "\"experience\": [\"Company — Role — Duration\", ...], "
        "\"education\": [...], \"certifications\": [...]}\nOnly JSON."
    )
    data = call_llm_json(prompt, max_tokens=1500)
    data["_meta"] = {"source": "linkedin-llm"}
    return data


def parse_linkedin_pdf(pdf_bytes: bytes, use_llm: bool = False) -> dict:
    """Parse a LinkedIn PDF export."""
    text = ""
    try:
        import pdfplumber, io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: text += t + "\n"
    except ImportError:
        try:
            from pypdf import PdfReader
            import io
            for page in PdfReader(io.BytesIO(pdf_bytes)).pages:
                t = page.extract_text()
                if t: text += t + "\n"
        except ImportError:
            raise RuntimeError("pip install pdfplumber или pypdf")
    if not text.strip():
        raise RuntimeError("Не удалось извлечь текст из PDF")
    return parse_linkedin_text(text, use_llm=use_llm, _source="linkedin-pdf")


def parse_linkedin_text(text: str, *, use_llm: bool = False,
                         _source: str = "linkedin-text") -> dict:
    """Parse raw LinkedIn profile text."""
    if not text or not text.strip():
        return {"name":"","hard_skills":[],"soft_skills":[],"interests":[],
                "experience":[],"education":[],"certifications":[],
                "_meta": {"source": _source, "chars": 0}}

    if use_llm or os.getenv("LLM_API_KEY", "").strip():
        try:
            r = _parse_linkedin_llm(text)
            for k in ("experience","education","certifications"):
                r.setdefault(k, [])
            r["_meta"] = {"source": "linkedin-llm", "chars": len(text)}
            return r
        except Exception:
            pass

    name = _extract_name_from_text(text)
    hard, soft = _extract_skills_regex(text)
    sections = _extract_sections(text)
    interests = _extract_interests(text)
    return {"name": name, "hard_skills": hard, "soft_skills": soft,
            "interests": interests, **sections,
            "_meta": {"source": _source, "chars": len(text)}}


if __name__ == "__main__":
    sample = """John Doe\nSenior Backend at Google\n\nExperience\nGoogle — Senior Backend — 2022-Present\nPython, Go, Kubernetes, AWS.\n\nSkills\nPython, Docker, PostgreSQL, CI/CD, Git, System Design"""
    r = parse_linkedin_text(sample)
    print(f"Name: {r['name']}")
    print(f"Hard ({len(r['hard_skills'])}): {', '.join(r['hard_skills'])}")
    print(f"Source: {r['_meta']['source']}")
