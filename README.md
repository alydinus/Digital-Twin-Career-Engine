# 🎯 Digital Twin Career Engine

> **Final Generative AI Project** — система, которая анализирует навыки,
> предсказывает карьерные направления, генерирует персональный план развития
> и готовит к собеседованию — с помощью GenAI.

---

## 🤖 Почему это GenAI-проект?

Шесть генеративных компонентов, каждый работает в двух режимах:

| Компонент | Fallback (без ключа) | GenAI (с LLM_API_KEY) |
|---|---|---|
| **Resume Parser** | Regex по словарю 60+ навыков | LLM извлекает навыки из произвольного текста |
| **Job Parser** | Regex + паттерны секций вакансии | LLM структурирует JD → JSON |
| **Skill Matcher** | Exact overlap score + алиасы | TF-IDF cosine similarity (sklearn) |
| **Career Coach** | Шаблонный roadmap по сложности | LLM генерирует персональный план |
| **Interview Coach** | Банк 100+ вопросов по навыкам | LLM создаёт персональные вопросы |
| **Resource Finder** | Статический словарь 20+ навыков | LLM подбирает актуальные ресурсы |

---

## 🏗️ Архитектура (5 слоёв)

```
 Input: profile / resume text / job description
            │
            ▼
 ┌──────────────────────────────────────────────────────┐
 │  1. PLATFORM LAYER                                   │
 │     data/profile.json  ── шаблон профиля             │
 │     data/jobs.csv      ── 5 ролей × 8 навыков        │
 │     utils/resume_parser.py  ◄── LLM / Regex          │
 │     utils/job_parser.py     ◄── LLM / Regex          │
 └─────────────────────┬────────────────────────────────┘
                       │  структурированный профиль
                       ▼
 ┌──────────────────────────────────────────────────────┐
 │  2. MODEL LAYER                                      │
 │     model/predictor.py        ── overlap + алиасы    │
 │     model/semantic_matcher.py ── TF-IDF cosine       │
 └─────────────────────┬────────────────────────────────┘
                       │  top-N roles + scores + missing
                       ▼
 ┌──────────────────────────────────────────────────────┐
 │  3. AGENT LAYER  (GenAI core)                        │
 │     agent/resource_finder.py ── курсы / docs / repo  │
 │     agent/career_coach.py    ── roadmap генератор    │
 │     agent/interview_coach.py ── вопросы к интервью   │
 └─────────────────────┬────────────────────────────────┘
                       │  ресурсы + план + вопросы
                       ▼
 ┌──────────────────────────────────────────────────────┐
 │  4. APPLICATION LAYER — Streamlit UI (6 вкладок)     │
 │     🎯 Predict · 🗺️ Roadmap · 🎤 Interview           │
 │     💼 Job Match · 📄 Resume · ℹ️ About              │
 └──────────────────────────────────────────────────────┘
                       │
 ┌──────────────────────────────────────────────────────┐
 │  5. INFRASTRUCTURE LAYER                             │
 │     Python 3.10 · venv · локально / Streamlit Cloud  │
 │     .env ── LLM_API_KEY, LLM_PROVIDER                │
 └──────────────────────────────────────────────────────┘
```

---

## 📁 Структура проекта

```
genAi/
├── data/
│   ├── profile.json          # профиль пользователя (шаблон / fallback)
│   └── jobs.csv              # база профессий: 5 ролей, 8 навыков каждая
│
├── model/
│   ├── predictor.py          # overlap score, нормализация, алиасы (k8s→kubernetes)
│   └── semantic_matcher.py   # TF-IDF cosine similarity + hybrid score
│
├── agent/
│   ├── resource_finder.py    # 20+ навыков → docs/курсы/GitHub (LLM + static)
│   ├── career_coach.py       # поэтапный roadmap по фазам (LLM + template)
│   └── interview_coach.py    # банк 100+ вопросов по навыкам (LLM + static)
│
├── utils/
│   ├── resume_parser.py      # текст резюме → структурированный профиль
│   └── job_parser.py         # текст вакансии → навыки + gap analysis
│
├── app/
│   └── app.py                # Streamlit UI — 6 вкладок
│
├── tests/
│   └── test_predictor.py     # 29 unit-тестов
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Быстрый старт

```bash
git clone <repo> && cd genAi
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

**Включить GenAI-режим (опционально):**
```bash
cp .env.example .env
# Отредактируй .env: LLM_API_KEY=sk-ant-...
# Или введи ключ прямо в боковой панели Streamlit
```

---

## 🎬 Демо-сценарий (защита проекта)

| # | Вкладка | Действие | Что показывает |
|---|---------|----------|----------------|
| 1 | **Resume** | Вставить текст резюме → кнопка | AI извлекает навыки из свободного текста |
| 2 | **Predict** | Кнопка Predict | Bar chart + radar chart + топ-3 роли |
| 3 | **Predict** | Раскрыть роль → выбрать навык | Ресурсы: docs / курсы / GitHub / видео |
| 4 | **Roadmap** | Выбрать роль → Сгенерировать | Поэтапный план: фазы, сроки, quick wins |
| 5 | **Interview** | Выбрать роль → Вопросы | Тех. вопросы по пробелам + behavioral |
| 6 | **Job Match** | Вставить вакансию → Анализ | Fit level, you_have / you_need, ресурсы |
| 7 | *(бонус)* | Ввести API-ключ → повторить | Сравнить LLM vs fallback вживую |

---

## ⚙️ Алгоритмы матчинга

### 1. Overlap Score
```
score = |user_skills ∩ role_skills| / |role_skills|
```
Алиасы: `k8s→kubernetes`, `torch→pytorch`, `ML→machine learning` и др.

### 2. Semantic (TF-IDF Cosine)
```python
tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
sim   = cosine_similarity(user_vecs, role_vecs)
# "Data Analysis" ≈ "Statistics", "ML" ≈ "Machine Learning"
```

### 3. Hybrid Score
```
hybrid = 0.6 × overlap + 0.4 × semantic
```

---

## 🤖 GenAI Pipeline

```
 Текст резюме / вакансии
        │
        ▼  LLM prompt → JSON
 Structured Profile / Job Requirements
        │
        ▼  Model Layer
 Top-N Roles + Scores + Missing Skills
        │
        ├──▶ Career Coach ──── LLM: "составь roadmap" → фазы + сроки
        ├──▶ Interview Coach ─ LLM: "создай вопросы" → тех + behavioral
        └──▶ Resource Finder ─ LLM: "найди ресурсы" → docs/курсы/GitHub
```

---

## 📊 Пример вывода

**Predict (50% match):**
```
🥇 Backend Engineer  50%  ✅ Python SQL Docker Git
                          ❌ System Design Microservices REST API PostgreSQL

🥈 DevOps Engineer   50%  ✅ Docker Linux Bash Git
                          ❌ Kubernetes AWS CI/CD Terraform

🥉 ML Engineer       50%  ✅ Python Docker SQL Git
                          ❌ PyTorch MLOps Math Scikit-learn
```

**Roadmap (DevOps, template):**
```
⚡ Быстрые победы: Bash, Linux, Git
📅 Фаза 1: CI/CD            — 2 нед.
📅 Фаза 2: Terraform, AWS   — 9 нед.
📅 Фаза 3: Kubernetes       — 4 нед.
⏱  Итого: ~3.8 мес.
```

**Interview Coach (пример вопросов):**
```
[Kubernetes] Объясни разницу между Pod, Deployment и Service.
             💡 Abstraction layers, scaling, routing

[System Design] Как бы ты спроектировал URL shortener?
                💡 Hashing, DB choice, caching, scalability
```

**Job Match:**
```
🟡 Senior Backend Engineer · Senior
   Fit: 🟠 Partial fit — 50%
   ✅ Есть:  Python, Docker, Git
   ❌ Нужно: PostgreSQL, Kubernetes, Microservices, CI/CD
```

---

## 🧪 Тестирование

```bash
pytest tests/ -v                        # все тесты
python model/predictor.py               # CLI: предсказание
python agent/interview_coach.py         # CLI: вопросы интервью
python utils/job_parser.py              # CLI: парсинг вакансии
```

**29 unit-тестов:** нормализация · score · missing skills · интеграция с реальными данными.


---

## 📄 PDF Resume Parser

Загрузи PDF-файл резюме прямо в интерфейс:

```
PDF файл
   │
   ▼  pdfplumber / pypdf / pdfminer (по приоритету)
Извлечённый текст
   │
   ▼  resume_parser.py (LLM / Regex)
Структурированный профиль → profile.json
```

Установка (выбери одну):
```bash
pip install pdfplumber    # рекомендуется
pip install pypdf         # fallback
```

---

## 📡 Telegram Job Search

Парсит IT-каналы Telegram и матчит вакансии с твоим профилем.

### Поддерживаемые каналы (10 штук)
`@getmatch` · `@python_jobs_ru` · `@devops_jobs_ru` · `@data_jobs_ru`
`@frontend_jobs` · `@backend_jobs_ru` · `@remote_jobs_ru`
`@hh_it_jobs` · `@tproger_jobs` · `@itjobs_ru`

### Режимы работы

**Mock-режим** (без настройки) — 8 демо-вакансий с реальной структурой и матчингом.

**Реальный поиск** — нужны Telegram API credentials:
```bash
# .env
TELEGRAM_API_ID=12345678       # https://my.telegram.org/apps
TELEGRAM_API_HASH=abcdef...
TELEGRAM_PHONE=+7XXXXXXXXXX
```
```bash
pip install telethon
```

### Telegram Bot

Управление системой прямо из Telegram:

```
/start   — инструкция
/predict — топ-3 профессии по навыкам
/jobs    — свежие вакансии по профилю
/profile — текущий профиль
/skills Python, Docker, SQL — обновить навыки

📄 Отправь PDF — автоматически обновит профиль
💬 Отправь текст резюме — то же самое
```

Настройка:
```bash
# .env
TELEGRAM_BOT_TOKEN=your_token   # @BotFather → /newbot

pip install python-telegram-bot
python telegram_bot/bot.py
```

---

## 🗂️ Полная структура проекта

```
genAi/
├── data/
│   ├── profile.json              # профиль пользователя
│   ├── jobs.csv                  # 5 базовых ролей
│   └── telegram_channels.json   # 10 IT-каналов Telegram
│
├── model/
│   ├── predictor.py              # overlap score + алиасы
│   └── semantic_matcher.py      # TF-IDF cosine similarity
│
├── agent/
│   ├── resource_finder.py        # курсы / docs / GitHub
│   ├── career_coach.py           # GenAI roadmap
│   └── interview_coach.py       # банк 100+ вопросов
│
├── utils/
│   ├── resume_parser.py          # текст → профиль (LLM / Regex)
│   ├── job_parser.py             # вакансия → навыки + gap
│   └── pdf_parser.py            # PDF → текст → профиль
│
├── telegram_bot/
│   ├── bot.py                   # Telegram бот (/jobs, /predict, PDF)
│   └── job_scraper.py           # парсер IT-каналов + mock
│
├── app/
│   └── app.py                   # Streamlit UI (8 вкладок)
│
├── tests/
│   └── test_predictor.py        # 29 unit-тестов
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🎬 Демо-сценарий (финальная версия)

| # | Вкладка | Действие | Эффект |
|---|---------|----------|--------|
| 1 | **PDF Upload** | Загрузить PDF резюме | Авто-извлечение навыков |
| 2 | **Predict** | Кнопка Predict | Bar + Radar chart, топ-3 роли |
| 3 | **Roadmap** | Выбрать роль → Generate | Поэтапный AI-план |
| 4 | **Interview** | Выбрать роль → Вопросы | 10+ вопросов по пробелам |
| 5 | **Job Match** | Вставить вакансию | Fit score + you_need |
| 6 | **Telegram Jobs** | Кнопка Найти | Топ вакансий из каналов |
| 7 | *(бонус)* | Ввести API-ключ | LLM vs fallback живьём |
| 8 | *(бонус)* | Запустить бота | Demo через Telegram |

---

## 🗺️ Roadmap проекта

### ✅ Реализовано
- `predictor.py` — overlap score + нормализация + 11 алиасов
- `semantic_matcher.py` — TF-IDF cosine + hybrid score
- `resource_finder.py` — 20+ навыков, 4 типа ресурсов (LLM + static)
- `career_coach.py` — roadmap по фазам с quick wins (LLM + template)
- `interview_coach.py` — 100+ вопросов по 15 навыкам (LLM + static)
- `resume_parser.py` — парсинг текста резюме (LLM + regex)
- `job_parser.py` — парсинг вакансии + gap analysis (LLM + regex)
- `pdf_parser.py` — PDF резюме → профиль (pdfplumber / pypdf / pdfminer)
- `job_scraper.py` — парсинг Telegram IT-каналов + mock (8 вакансий)
- `bot.py` — Telegram бот (/jobs, /predict, PDF upload)
- `app.py` — Streamlit UI: 8 вкладок, bar chart, radar chart
- Экспорт отчёта в Markdown
- 29 unit-тестов

### 🔜 Следующие шаги
- [ ] PDF-парсер резюме (`pdfplumber`)
- [ ] LinkedIn/HH.ru scraper
- [ ] Расширение `jobs.csv` до 20+ ролей
- [ ] Векторная БД (ChromaDB) для семантического поиска
- [ ] История сессий / прогресс-трекер
- [ ] Деплой на Streamlit Community Cloud
- [ ] Dockerfile

---

## 🛠️ Стек

| Слой | Технологии |
|------|-----------|
| Language | Python 3.10 |
| UI | Streamlit |
| Charts | Matplotlib (radar), Streamlit bar chart |
| Data | Pandas, CSV, JSON |
| ML | scikit-learn (TF-IDF, cosine similarity) |
| GenAI | Anthropic Claude / OpenAI GPT (опционально) |
| Telegram | Telethon (scraper) + python-telegram-bot (бот) |
| PDF | pdfplumber / pypdf / pdfminer |
| Testing | pytest / unittest |
| Config | python-dotenv |

---

## 👤 Автор

**Alydin** · Final Generative AI Project · 2026
