"""
Utils — Resume Parser v2

Режимы:
  1. Regex fallback  — 150+ навыков, section-aware парсинг, кириллица
  2. LLM mode        — если задан LLM_API_KEY

Улучшения:
  - 150+ hard skills (backend, devops, data, frontend, mobile, cloud)
  - Section-aware: ищет навыки в секциях Skills/Стек/Технологии
  - Кириллические имена
  - Обогащение LLM-результата regex при скудном выводе
  - Исправлен баг provider в LLM-режиме
"""

from __future__ import annotations
import os
import re
from utils.llm_client import call_llm_json, is_available

# ---------------------------------------------------------------------------
# Словари навыков
# ---------------------------------------------------------------------------

KNOWN_HARD_SKILLS = [
    # Languages
    "Python","Java","JavaScript","TypeScript","Go","Golang","Rust","C++","C#","C",
    "PHP","Ruby","Kotlin","Swift","Scala","Elixir","Clojure","Haskell","R","MATLAB",
    "Dart","Lua","Perl","Shell","PowerShell",
    # Web Frameworks
    "FastAPI","Django","Flask","Spring","Spring Boot","Express","NestJS",
    "Next.js","Nuxt.js","Laravel","Rails","ASP.NET","Gin","Echo","Fiber",
    # Frontend
    "React","Vue","Angular","Svelte","HTML","CSS","SASS","Tailwind","Bootstrap",
    "jQuery","Webpack","Vite","Redux","MobX",
    # Databases
    "SQL","PostgreSQL","MySQL","SQLite","MariaDB","MongoDB","Redis",
    "Elasticsearch","Cassandra","DynamoDB","CouchDB","Neo4j","InfluxDB",
    "ClickHouse","Snowflake","BigQuery","Redshift",
    # DevOps & Infra
    "Docker","Kubernetes","k8s","Helm","Terraform","Ansible","Puppet","Chef",
    "Vagrant","Nginx","Apache","HAProxy","Traefik",
    "CI/CD","GitHub Actions","GitLab CI","Jenkins","CircleCI","ArgoCD",
    # Cloud
    "AWS","GCP","Azure","DigitalOcean","Heroku","Cloudflare","Vercel","Netlify",
    "S3","EC2","Lambda","EKS","ECS","RDS","Cloud Functions","Cloud Run",
    # Monitoring
    "Prometheus","Grafana","Datadog","New Relic","ELK","Loki",
    "Jaeger","Zipkin","OpenTelemetry",
    # Messaging
    "Kafka","RabbitMQ","NATS","Celery","SQS",
    # API
    "REST API","GraphQL","gRPC","WebSocket","OpenAPI","Swagger",
    # Data / ML
    "Pandas","NumPy","Matplotlib","Seaborn","Plotly","Scikit-learn",
    "PyTorch","TensorFlow","Keras","XGBoost","LightGBM","CatBoost",
    "Hugging Face","LangChain","Jupyter","Spark","PySpark","Airflow",
    "dbt","MLflow","MLOps","DVC",
    # Mobile
    "Android","iOS","React Native","Flutter",
    # Tools & Practices
    "Git","Linux","Bash","Makefile","Microservices","System Design",
    "DDD","TDD","BDD","Agile","Scrum","Kanban","Jira","Confluence",
    "Figma","Postman","OAuth","JWT",
]

KNOWN_SOFT_SKILLS = [
    "problem solving","teamwork","communication","leadership",
    "critical thinking","time management","adaptability","creativity",
    "collaboration","mentoring","presentation","decision making",
    "self-management","proactivity","ownership","empathy",
    "conflict resolution","negotiation","coaching",
    # Russian
    "решение проблем","командная работа","коммуникация","лидерство",
    "критическое мышление","управление временем","адаптивность","наставничество",
]

KNOWN_INTERESTS = [
    "backend","frontend","fullstack","devops","data","AI","ML",
    "cloud","security","automation","open source","mobile",
    "blockchain","embedded","game development","fintech","edtech",
    "healthcare","web3","microservices","distributed systems",
    # Russian
    "бэкенд","фронтенд","данные","безопасность","автоматизация",
]

# Заголовки секций с навыками
_SECTION_RE = re.compile(
    r"(?:^|\n)[ \t]*"
    r"(?:(?:hard\s+)?skills?|tech(?:nical)?\s*skills?|technologies|stack"
    r"|(?:ключевые\s+)?навыки|технологии|инструменты|компетенции"
    r"|опыт\s+работы\s+с|tools?\s+(?:and|&)\s+tech|знания\s+и\s+навыки)"
    r"\s*[:\-–—]?\s*(?:\n|$)"
    r"([\s\S]{0,800}?)"
    r"(?=\n[ \t]*[A-ZА-ЯЀ-ӿ][^\n]{0,40}[:\-–—]|\Z)",
    re.IGNORECASE,
)


def _extract_name(text: str) -> str:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return "User"

    marker_re = re.compile(
        r"(?:^name|^имя|^ф\.?\s*и\.?\s*о\.?)\s*[:\-–]?\s*"
        r"([A-ZА-Яa-zа-яёЁЀ-ӿ][a-zA-Zа-яА-ЯёЁЀ-ӿ\s]{2,40})",
        re.IGNORECASE | re.MULTILINE,
    )
    m = marker_re.search(text)
    if m:
        return m.group(1).strip()

    name_re = re.compile(
        r"^([A-ZА-ЯЁА-Я][a-zа-яёа-я]{1,20}"
        r"(?:\s+[A-ZА-ЯЁА-Я][a-zа-яёа-я]{1,20}){1,2})$"
    )
    for line in lines[:6]:
        clean = re.sub(r"[—\-–—|·•]", "", line).strip()
        if name_re.match(clean) and len(clean.split()) >= 2:
            return clean

    first = lines[0]
    if len(first.split()) <= 4 and len(first) < 50:
        return first
    return "User"


def _find_skills(text: str) -> list[str]:
    found = set()
    tl = text.lower()
    for skill in KNOWN_HARD_SKILLS:
        pat = r"(?<![a-zA-Z0-9+#/])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9+#/])"
        if re.search(pat, tl):
            found.add(skill)
    return sorted(found)


def _extract_by_regex(text: str) -> dict:
    text = re.sub(r"\r\n|\r", "\n", text)
    name = _extract_name(text)

    # Секционный поиск
    section_skills: list[str] = []
    for m in _SECTION_RE.finditer(text):
        section_skills.extend(_find_skills(m.group(1)))

    # Глобальный поиск по всему тексту
    global_skills = _find_skills(text)

    # Объединяем с приоритетом секционных
    hard = sorted(set(section_skills + global_skills))

    tl = text.lower()
    soft      = [s for s in KNOWN_SOFT_SKILLS if s.lower() in tl]
    interests = [s for s in KNOWN_INTERESTS   if s.lower() in tl]

    # Роль из заголовка → интерес
    role_re = re.compile(
        r"\b(backend|frontend|fullstack|devops|data\s+(?:scientist|engineer|analyst)"
        r"|ml\s+engineer|software\s+engineer|developer|architect|lead)\b",
        re.IGNORECASE,
    )
    for line in text.splitlines()[:8]:
        m = role_re.search(line)
        if m:
            ri = m.group(1).lower().strip()
            if ri not in [i.lower() for i in interests]:
                interests.append(ri)

    return {
        "name": name,
        "hard_skills": hard,
        "soft_skills": list(dict.fromkeys(soft)),
        "interests":   list(dict.fromkeys(interests)),
        "_meta": {"source": "resume_parser_regex"},
    }


def _extract_by_llm(text: str) -> dict:
    if not is_available():
        raise ValueError("LLM_API_KEY не задан")

    provider = os.getenv("LLM_PROVIDER", "unknown")  # fix: определяем переменную

    prompt = f"""
Ты — точный парсер резюме. Верни ТОЛЬКО валидный JSON без пояснений:
{{
  "name": "полное имя кандидата",
  "hard_skills": ["СПИСОК ВСЕХ технических навыков: языки, фреймворки, БД, инструменты, облака, DevOps"],
  "soft_skills": ["мягкие навыки"],
  "interests": ["профессиональные интересы и области"]
}}

Правила:
- Включай ВСЕ технологии из текста (и в описании опыта, и в секции навыков)
- hard_skills — только конкретные технологии, не общие фразы
- Без дублей

Текст резюме:
\"\"\"{text[:4000]}\"\"\"
""".strip()

    data = call_llm_json(prompt, max_tokens=1500)
    data["_meta"] = {"source": "resume_parser_llm", "provider": provider}
    return data


def parse_resume(text: str, use_llm: bool = False) -> dict:
    if not text or not text.strip():
        return {"name": "User", "hard_skills": [], "soft_skills": [],
                "interests": [], "_meta": {"source": "empty"}}

    if use_llm or os.getenv("LLM_API_KEY"):
        try:
            result = _extract_by_llm(text)
            # Обогащаем если LLM вернул мало навыков
            if len(result.get("hard_skills", [])) < 4:
                regex_r = _extract_by_regex(text)
                merged  = list(dict.fromkeys(
                    result.get("hard_skills", []) + regex_r["hard_skills"]
                ))
                result["hard_skills"] = merged
                result["_meta"]["source"] += "+regex_enriched"
            return result
        except Exception as e:
            print(f"[ResumeParser] LLM недоступен ({e}), regex fallback.")

    return _extract_by_regex(text)


if __name__ == "__main__":
    import json, sys
    sample = """
    Иван Петров — Senior Backend Developer
    Стек: Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, AWS, Kafka, Celery
    Также: Terraform, GitHub Actions, Grafana, Prometheus
    Soft: problem solving, teamwork, leadership
    Интересы: backend, cloud, distributed systems
    """
    t = open(sys.argv[1]).read() if len(sys.argv) > 1 else sample
    r = parse_resume(t)
    print(json.dumps({k: v for k, v in r.items() if not k.startswith("_")},
                     indent=2, ensure_ascii=False))
    print(f"Источник: {r['_meta']['source']} | Hard skills: {len(r['hard_skills'])}")
