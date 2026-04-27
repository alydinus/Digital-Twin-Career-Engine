"""
Telegram Job Scraper

Парсит IT-каналы Telegram в поиске вакансий и матчит их с профилем пользователя.

Требования:
  pip install telethon python-dotenv

Настройка (.env):
  TELEGRAM_API_ID=12345678
  TELEGRAM_API_HASH=abcdef1234567890abcdef
  TELEGRAM_PHONE=+7XXXXXXXXXX   # или TELEGRAM_BOT_TOKEN для бота

Получить API ID/Hash: https://my.telegram.org/apps

Архитектура:
  1. TelegramScraper  — подключается к Telegram, читает каналы
  2. JobExtractor     — извлекает навыки из текста вакансии
  3. JobMatcher       — матчит вакансии с профилем пользователя
  4. MockScraper      — fallback с демо-данными (без Telegram)
"""

from __future__ import annotations
import os
import re
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Dataclass для вакансии
# ---------------------------------------------------------------------------

@dataclass
class JobPost:
    id:             int
    channel:        str
    channel_name:   str
    text:           str
    date:           datetime
    url:            str
    required_skills: list[str] = field(default_factory=list)
    role_name:      str = ""
    seniority:      str = "middle"
    match_score:    float = 0.0
    match_pct:      int   = 0
    you_have:       list[str] = field(default_factory=list)
    you_need:       list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "channel": self.channel,
            "channel_name": self.channel_name,
            "text": self.text[:500] + ("..." if len(self.text) > 500 else ""),
            "date": self.date.isoformat(),
            "url": self.url,
            "required_skills": self.required_skills,
            "role_name": self.role_name,
            "seniority": self.seniority,
            "match_score": self.match_score,
            "match_pct": self.match_pct,
            "you_have": self.you_have,
            "you_need": self.you_need,
        }


# ---------------------------------------------------------------------------
# Извлечение навыков и информации из поста
# ---------------------------------------------------------------------------

with open(ROOT / "data" / "telegram_channels.json", encoding="utf-8") as _f:
    _CHANNELS_CONFIG = json.load(_f)

_SENIORITY_RE = {
    "senior":  re.compile(r"\b(senior|lead|principal|staff|architect|sr\.?)\b", re.I),
    "junior":  re.compile(r"\b(junior|intern|jr\.?|entry.?level|graduate)\b", re.I),
}

def _detect_seniority(text: str) -> str:
    for level, pattern in _SENIORITY_RE.items():
        if pattern.search(text):
            return level
    return "middle"


def _extract_role(text: str) -> str:
    """Пытается найти название роли в первых 3 строках поста."""
    for line in text.strip().split("\n")[:4]:
        line = line.strip().strip("*#_📌🔥💼")
        if 5 < len(line) < 80 and any(w in line.lower() for w in [
            "engineer", "developer", "scientist", "analyst", "devops",
            "architect", "manager", "lead", "intern",
            "инженер", "разработчик", "аналитик", "специалист",
        ]):
            return line
    return "IT Position"


def extract_job_info(text: str) -> dict:
    """Извлекает структурированную информацию из текста вакансии."""
    from utils.resume_parser import KNOWN_HARD_SKILLS
    text_lower = text.lower()
    skills = [s for s in KNOWN_HARD_SKILLS if s.lower() in text_lower]
    return {
        "role_name":       _extract_role(text),
        "seniority":       _detect_seniority(text),
        "required_skills": skills,
    }


# ---------------------------------------------------------------------------
# Матчинг вакансий с профилем
# ---------------------------------------------------------------------------

def match_job_to_profile(job: JobPost, profile: dict) -> JobPost:
    """Вычисляет match score вакансии относительно профиля."""
    from model.predictor import normalize_skill

    user_norm = profile.get("_all_skills_norm", set())
    if not user_norm:
        from model.predictor import normalize_skills
        user_norm = normalize_skills(
            profile.get("hard_skills", []) + profile.get("soft_skills", [])
        )

    req      = job.required_skills
    you_have = [s for s in req if normalize_skill(s) in user_norm]
    you_need = [s for s in req if normalize_skill(s) not in user_norm]
    score    = len(you_have) / len(req) if req else 0.0

    job.match_score = round(score, 4)
    job.match_pct   = int(score * 100)
    job.you_have    = you_have
    job.you_need    = you_need
    return job


# ---------------------------------------------------------------------------
# Mock scraper (fallback без Telegram)
# ---------------------------------------------------------------------------

_MOCK_JOBS = [
    {
        "id": 1001, "channel": "getmatch", "channel_name": "getmatch",
        "date_offset": 1,
        "text": """🔥 Senior Backend Engineer | Fintech | Remote

Компания: PayTech (серия B)
Формат: Remote / Москва

Стек: Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, Kafka

Требования:
- Python 5+ лет
- PostgreSQL, Redis
- Docker, Kubernetes
- Опыт с Kafka или RabbitMQ
- REST API, Microservices

Будет плюсом: AWS, Terraform

ЗП: 350 000 – 500 000 ₽
Контакт: @recruiter_paytech""",
    },
    {
        "id": 1002, "channel": "devops_jobs_ru", "channel_name": "DevOps Jobs RU",
        "date_offset": 0,
        "text": """💼 DevOps Engineer | Стартап | Гибрид

Ищем DevOps в растущий стартап (SaaS продукт).

Обязательно:
- Docker, Kubernetes (k8s)
- CI/CD: GitLab CI или GitHub Actions
- Linux, Bash
- AWS или GCP

Опционально: Terraform, Helm, Prometheus

ЗП: 250 000 – 350 000 ₽
Откликнуться: t.me/devops_hr""",
    },
    {
        "id": 1003, "channel": "python_jobs_ru", "channel_name": "Python Jobs RU",
        "date_offset": 2,
        "text": """🐍 Middle Python Developer | E-commerce | Офис/Remote

Стек: Python, Django, PostgreSQL, Redis, Docker, Git, Celery

Задачи:
- Разработка API на Django REST Framework
- Оптимизация запросов к PostgreSQL
- Поддержка CI/CD пайплайнов

Требования:
- Python 3+ года, Django
- SQL, PostgreSQL
- Docker, Git

ЗП: 200 000 – 280 000 ₽""",
    },
    {
        "id": 1004, "channel": "data_jobs_ru", "channel_name": "Data Science Jobs",
        "date_offset": 1,
        "text": """📊 ML Engineer | AI Product | Remote

Мы строим LLM-powered продукт, ищем ML Engineer.

Hard requirements:
- Python, PyTorch или TensorFlow
- MLOps: MLflow, DVC или аналоги
- Docker, Git
- SQL, опыт с данными

Nice to have:
- Опыт с LLM / fine-tuning
- Kubernetes для деплоя моделей
- AWS SageMaker

ЗП: $4000 – $6000 / мес""",
    },
    {
        "id": 1005, "channel": "backend_jobs_ru", "channel_name": "Backend Jobs RU",
        "date_offset": 3,
        "text": """🚀 Junior Backend Developer | EdTech | Офис (Москва)

Отличная возможность для начинающих!

Стек: Python, FastAPI, PostgreSQL, Docker, Git

Требования:
- Python (знание основ)
- SQL базовый уровень
- Git
- Желание учиться!

Обучаем: Docker, FastAPI, PostgreSQL
ЗП: 120 000 – 160 000 ₽
HR: @edtech_hr_ru""",
    },
    {
        "id": 1006, "channel": "remote_jobs_ru", "channel_name": "Remote Jobs RU",
        "date_offset": 0,
        "text": """🌍 Full Stack Engineer | SaaS | Полный Remote

Команда: 5 разработчиков, стартап из Европы.

Backend: Python, FastAPI, PostgreSQL
Frontend: React, TypeScript
Infra: Docker, AWS, GitHub Actions (CI/CD)

Требования:
- Python + React
- PostgreSQL, REST API
- Docker, Git, CI/CD

ЗП: €3500 – €5000 / мес
Английский: B2+""",
    },
    {
        "id": 1007, "channel": "tproger_jobs", "channel_name": "Tproger Jobs",
        "date_offset": 4,
        "text": """⚡ Backend Engineer (Go/Python) | Highload | Гибрид

Продукт: 10M+ пользователей, highload.

Стек: Go или Python, PostgreSQL, Redis, Kafka, Docker, Kubernetes

Требования:
- Go или Python (senior level)
- System Design, Microservices
- PostgreSQL, Redis
- Kafka / RabbitMQ
- Docker, Kubernetes

ЗП: 400 000 – 600 000 ₽""",
    },
    {
        "id": 1009, "channel": "dev_kg", "channel_name": "Dev KG",
        "date_offset": 0,
        "text": """💻 Python Backend Developer | Бишкек / Remote

Компания занимается разработкой fintech-решений для рынка Центральной Азии.

Требования:
- Python (Django / FastAPI)
- PostgreSQL, Redis
- Docker, Git
- REST API

Будет плюсом:
- Kubernetes, AWS
- Опыт с Celery, RabbitMQ

Формат: офис Бишкек или Remote
ЗП: договорная (рыночная)
📩 Контакт: @dev_kg_hr""",
    },
    {
        "id": 1010, "channel": "dev_kg", "channel_name": "Dev KG",
        "date_offset": 1,
        "text": """🔧 DevOps Engineer | IT-компания | Бишкек

Стек: Linux, Docker, Kubernetes, GitLab CI, Ansible

Требования:
- Linux, Bash — обязательно
- Docker, CI/CD (GitLab CI или GitHub Actions)
- Kubernetes — будет плюсом
- Git

ЗП: $1500–$2500
📩 @devops_kg_jobs""",
    },
    {
        "id": 1011, "channel": "dev_kg", "channel_name": "Dev KG",
        "date_offset": 2,
        "text": """📊 Data Analyst | Стартап | Remote

Ищем аналитика данных в быстрорастущий стартап.

Требования:
- Python (Pandas, NumPy)
- SQL (PostgreSQL)
- Jupyter, Matplotlib или Seaborn
- Git

Плюсом: Tableau, Power BI, Airflow

ЗП: $1000–$1800 / remote
📩 t.me/data_hr_kg""",
    },
    {
        "id": 1008, "channel": "hh_it_jobs", "channel_name": "HH IT Jobs",
        "date_offset": 2,
        "text": """🔧 Platform Engineer | Банк | Remote-first

Задачи:
- Поддержка и развитие Kubernetes кластеров
- Автоматизация через Terraform + Ansible
- CI/CD на GitLab CI
- Мониторинг: Prometheus, Grafana

Требования:
- Linux, Bash, Docker, Kubernetes
- Terraform, CI/CD
- AWS или On-premise
- Git

ЗП: 300 000 – 420 000 ₽""",
    },
]



def add_custom_channel(username: str, name: str = "", tags: list[str] | None = None) -> bool:
    """
    Добавляет кастомный канал в конфиг telegram_channels.json.

    Args:
        username: @username канала (без @)
        name:     отображаемое имя (по умолчанию = username)
        tags:     список тегов для фильтрации

    Returns:
        True если добавлен, False если уже был
    """
    username = username.lstrip("@").strip()
    if not username:
        return False

    cfg_path = ROOT / "data" / "telegram_channels.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    existing = {c["username"] for c in cfg["channels"]}
    if username in existing:
        return False

    cfg["channels"].append({
        "username": username,
        "name":     name or username,
        "tags":     tags or ["backend", "frontend", "devops", "data"],
    })

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[Channels] Добавлен канал: @{username}")
    return True


def get_mock_jobs(profile: dict, days_back: int = 7, min_match: int = 0) -> list[JobPost]:
    """
    Возвращает mock-вакансии с матчингом по профилю.
    Используется как fallback когда Telegram недоступен.
    """
    now  = datetime.now()
    jobs = []

    for raw in _MOCK_JOBS:
        info = extract_job_info(raw["text"])
        job  = JobPost(
            id           = raw["id"],
            channel      = raw["channel"],
            channel_name = raw["channel_name"],
            text         = raw["text"],
            date         = now - timedelta(days=raw.get("date_offset", 0)),
            url          = f"https://t.me/{raw['channel']}/{raw['id']}",
            required_skills = info["required_skills"],
            role_name    = info["role_name"],
            seniority    = info["seniority"],
        )
        job = match_job_to_profile(job, profile)
        if job.match_pct >= min_match:
            jobs.append(job)

    jobs.sort(key=lambda j: j.match_score, reverse=True)
    return jobs


# ---------------------------------------------------------------------------
# Telegram Scraper (требует telethon)
# ---------------------------------------------------------------------------

class TelegramJobScraper:
    """
    Парсит вакансии из Telegram-каналов через Telethon.

    Настройка:
      TELEGRAM_API_ID   — получить на https://my.telegram.org/apps
      TELEGRAM_API_HASH — там же
      TELEGRAM_PHONE    — номер телефона (+7...)
    """

    def __init__(self):
        self.api_id   = os.getenv("TELEGRAM_API_ID")
        self.api_hash = os.getenv("TELEGRAM_API_HASH")
        self.phone    = os.getenv("TELEGRAM_PHONE")
        self.client   = None

    def is_configured(self) -> bool:
        """True только если есть все ключи И сессия уже создана (авторизована)."""
        session_file = ROOT / "data" / ".telegram_session.session"
        return bool(self.api_id and self.api_hash and session_file.exists())

    async def connect(self):
        """
        Подключается к Telegram используя сохранённую сессию.
        Сессия создаётся один раз через: python telegram_bot/auth.py
        """
        try:
            from telethon import TelegramClient
        except ImportError:
            raise RuntimeError("Установи telethon: pip install telethon")

        session_file = ROOT / "data" / ".telegram_session.session"
        if not session_file.exists():
            raise RuntimeError(
                "Telegram сессия не найдена. "
                "Запусти один раз: python telegram_bot/auth.py"
            )

        session_path = ROOT / "data" / ".telegram_session"
        self.client  = TelegramClient(str(session_path), self.api_id, self.api_hash)
        # connect() — без интерактивного запроса OTP, только существующая сессия
        await self.client.connect()
        if not await self.client.is_user_authorized():
            raise RuntimeError(
                "Сессия устарела или невалидна. "
                "Удали data/.telegram_session.session и запусти снова: python telegram_bot/auth.py"
            )

    async def _fetch_channel(
        self,
        channel_username: str,
        limit: int = 50,
        days_back: int = 7,
    ) -> list[dict]:
        """Читает последние посты из канала."""
        from telethon.errors import ChannelPrivateError, UsernameNotOccupiedError
        cutoff = datetime.now() - timedelta(days=days_back)
        posts  = []

        try:
            entity = await self.client.get_entity(channel_username)
            async for msg in self.client.iter_messages(entity, limit=limit):
                if not msg.text or msg.date.replace(tzinfo=None) < cutoff:
                    continue
                posts.append({
                    "id":      msg.id,
                    "text":    msg.text,
                    "date":    msg.date.replace(tzinfo=None),
                    "channel": channel_username,
                    "url":     f"https://t.me/{channel_username}/{msg.id}",
                })
        except (ChannelPrivateError, UsernameNotOccupiedError) as e:
            print(f"[TelegramScraper] Канал {channel_username} недоступен: {e}")
        except Exception as e:
            print(f"[TelegramScraper] Ошибка {channel_username}: {e}")

        return posts

    async def fetch_jobs(
        self,
        profile: dict,
        channels: list[str] | None = None,
        days_back: int = 7,
        limit_per_channel: int = 50,
        min_match_pct: int = 20,
    ) -> list[JobPost]:
        """
        Основной метод: парсит каналы и возвращает отсортированные вакансии.

        Args:
            profile:            профиль пользователя
            channels:           список @username каналов (None = все из конфига)
            days_back:          глубина поиска в днях
            limit_per_channel:  макс. постов с канала
            min_match_pct:      минимальный match score для отображения

        Returns:
            list[JobPost] отсортированный по match_score desc
        """
        if channels is None:
            channels = [c["username"] for c in _CHANNELS_CONFIG["channels"]]

        all_posts = []
        for ch in channels:
            posts = await self._fetch_channel(ch, limit=limit_per_channel, days_back=days_back)
            all_posts.extend(posts)

        print(f"[TelegramScraper] Собрано {len(all_posts)} постов из {len(channels)} каналов")

        # Определяем channel_name
        ch_names = {c["username"]: c["name"] for c in _CHANNELS_CONFIG["channels"]}

        jobs = []
        for post in all_posts:
            info = extract_job_info(post["text"])
            if not info["required_skills"]:
                continue  # пропускаем нетехнические посты

            job = JobPost(
                id           = post["id"],
                channel      = post["channel"],
                channel_name = ch_names.get(post["channel"], post["channel"]),
                text         = post["text"],
                date         = post["date"],
                url          = post["url"],
                required_skills = info["required_skills"],
                role_name    = info["role_name"],
                seniority    = info["seniority"],
            )
            job = match_job_to_profile(job, profile)
            if job.match_pct >= min_match_pct:
                jobs.append(job)

        jobs.sort(key=lambda j: j.match_score, reverse=True)
        print(f"[TelegramScraper] {len(jobs)} вакансий после фильтрации (min {min_match_pct}%)")
        return jobs

    async def disconnect(self):
        if self.client:
            await self.client.disconnect()


# ---------------------------------------------------------------------------
# Удобная sync-обёртка
# ---------------------------------------------------------------------------

def scrape_jobs(
    profile: dict,
    channels: list[str] | None = None,
    days_back: int = 7,
    min_match_pct: int = 20,
    use_mock: bool = False,
) -> list[JobPost]:
    """
    Синхронная обёртка для парсинга вакансий.
    Автоматически выбирает mock или реальный Telegram.
    """
    scraper = TelegramJobScraper()

    if use_mock or not scraper.is_configured():
        print("[TelegramScraper] Использую mock-данные (настрой .env для реального Telegram)")
        return get_mock_jobs(profile, days_back=days_back, min_match=min_match_pct)

    async def _run():
        await scraper.connect()
        try:
            return await scraper.fetch_jobs(
                profile=profile,
                channels=channels,
                days_back=days_back,
                min_match_pct=min_match_pct,
            )
        finally:
            await scraper.disconnect()

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# CLI тест
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    from model.predictor import load_profile, normalize_skills

    profile = load_profile(ROOT / "data" / "profile.json")
    jobs    = scrape_jobs(profile, use_mock=True, min_match_pct=30)

    print(f"\n{'='*60}")
    print(f"Найдено вакансий: {len(jobs)}\n")
    for j in jobs:
        print(f"  {j.match_pct:3d}%  [{j.seniority:6s}]  {j.role_name}")
        print(f"         {j.channel_name} · {j.date.strftime('%d.%m')}")
        print(f"         ✅ {', '.join(j.you_have) or '—'}")
        print(f"         ❌ {', '.join(j.you_need) or '—'}")
        print(f"         🔗 {j.url}\n")
