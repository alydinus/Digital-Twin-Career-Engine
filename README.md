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
 Input: CV PDF / LinkedIn / NotebookLM / text
            │
            ▼
 ┌──────────────────────────────────────────────────────┐
 │  1. PLATFORM LAYER  (MCP Integration)                │
 │     utils/notebooklm_bridge.py ◄── MCP / manual      │
 │     utils/linkedin_parser.py   ◄── PDF / text / LLM  │
 │     utils/resume_parser.py     ◄── LLM / Regex       │
 │     utils/job_parser.py        ◄── LLM / Regex       │
 │     data/profile.json  ── шаблон профиля              │
 │     data/jobs.csv      ── 21 роль × 8 навыков         │
 └─────────────────────┬────────────────────────────────┘
                       │  структурированный профиль
                       ▼
 ┌──────────────────────────────────────────────────────┐
 │  2. MODEL LAYER  (ML Prediction)                     │
 │     model/predictor.py        ── overlap + алиасы     │
 │     model/semantic_matcher.py ── TF-IDF cosine        │
 └─────────────────────┬────────────────────────────────┘
                       │  top-N roles + scores + missing
                       ▼
 ┌──────────────────────────────────────────────────────┐
 │  3. AGENT LAYER  (GenAI + Web Browsing)               │
 │     agent/resource_finder.py ── LLM → Web → Static    │
 │     agent/web_searcher.py    ── DuckDuckGo + GitHub    │
 │     agent/career_coach.py    ── roadmap генератор      │
 │     agent/interview_coach.py ── вопросы к интервью     │
 │     agent/live_coach.py      ── chatbot (mentor/roast) │
 └─────────────────────┬────────────────────────────────┘
                       │  ресурсы + план + вопросы + чат
                       ▼
 ┌──────────────────────────────────────────────────────┐
 │  4. APPLICATION LAYER — Streamlit UI (8 вкладок)      │
 │     📄 CV Upload (PDF · Text · LinkedIn · NotebookLM) │
 │     🎯 Predict · 🔥 CV Roast · 📨 Telegram Jobs       │
 │     🗺️ Roadmap · 🎤 Interview · 💬 Live Coach          │
 │                                                       │
 │  Виджеты:                                             │
 │     ⚖️ Balance Wheel  ── radar Hard vs Soft skills     │
 │     🎮 RPG Tech Tree  ── locked / unlocked nodes       │
 │     🎁 Semester Wrapped ─ shareable LinkedIn PNG       │
 │     🤖 Live Coach     ── "Roast My Stack" toggle       │
 └──────────────────────────────────────────────────────┘
                       │
 ┌──────────────────────────────────────────────────────┐
 │  5. INFRASTRUCTURE LAYER                              │
 │     Python 3.10+ · venv · локально / Docker / Cloud   │
 │     .env ── LLM_API_KEY, GITHUB_TOKEN, NOTEBOOKLM_URL │
 └──────────────────────────────────────────────────────┘
```

---

## 🖥️ Infrastructure Layer — где что считается

PDF требует явно перечислить, что крутится локально на CPU, а что — в облаке.

| Компонент | Где работает | Нагрузка | Зависимости |
|---|---|---|---|
| Streamlit UI (`app/app.py`) | **Локально, CPU** | низкая | streamlit, matplotlib |
| Dashboard widgets (`app/dashboard.py`) | **Локально, CPU** | низкая (matplotlib) | matplotlib, numpy |
| Predictor — overlap (`model/predictor.py`) | **Локально, CPU** | O(N×M) ≈ ms | pandas |
| Semantic Matcher — TF-IDF (`model/semantic_matcher.py`) | **Локально, CPU** | O(N×M) ≈ ms | scikit-learn |
| Resume / PDF Parser (regex fallback) | **Локально, CPU** | низкая | pdfplumber / regex |
| LinkedIn Parser (`utils/linkedin_parser.py`) | **Локально, CPU** | низкая | pdfplumber / regex |
| CV Roast (rule-based) | **Локально, CPU** | мгновенно | — |
| Live Coach (rule-based) | **Локально, CPU** | мгновенно | — |
| Resource Finder (static dict) | **Локально, CPU** | мгновенно | — |
| Telegram Job Scraper (mock) | **Локально, CPU** | низкая | — |
| Telegram Job Scraper (Telethon) | **Локально + Telegram cloud** | сетевая | telethon, MTProto |
| Agent Web Browsing (`agent/web_searcher.py`) | **Локально → Internet** | сетевая | urllib (stdlib) |
| NotebookLM MCP Bridge (`utils/notebooklm_bridge.py`) | **Локально → Google Cloud** | API-вызов | notebooklm-mcp-server |
| GitHub Search API (Agent Layer) | **Локально → GitHub Cloud** | API-вызов | GITHUB_TOKEN |
| LLM-режим Resume Parser | **Cloud (Anthropic / OpenAI / Gemini)** | API-вызов | LLM_API_KEY |
| LLM-режим Career Coach | **Cloud (Anthropic / OpenAI / Gemini)** | API-вызов | LLM_API_KEY |
| LLM-режим Interview Coach | **Cloud (Anthropic / OpenAI / Gemini)** | API-вызов | LLM_API_KEY |
| LLM-режим Resource Finder | **Cloud (Anthropic / OpenAI / Gemini)** | API-вызов | LLM_API_KEY |
| LLM-режим CV Roast | **Cloud (Anthropic / OpenAI / Gemini)** | API-вызов | LLM_API_KEY |
| LLM-режим Live Coach | **Cloud (Anthropic / OpenAI / Gemini)** | API-вызов | LLM_API_KEY |

**Hosting вариант:**

* Локально: `python -m venv venv && streamlit run app/app.py` — всё на твоём CPU, LLM-фичи опциональны.
* Docker: `docker compose up -d` — изолированный контейнер, тот же CPU.
* Streamlit Community Cloud: бесплатный хостинг UI + ML, LLM-ключ задаётся через secrets.
* Cloud LLM endpoint всегда внешний (OpenAI / Anthropic). Никакая модель не запускается локально.

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
│   ├── jobs.csv                  # 21 роль × 8 навыков
│   └── telegram_channels.json   # 10 IT-каналов Telegram
│
├── model/
│   ├── predictor.py              # overlap score + алиасы
│   └── semantic_matcher.py      # TF-IDF cosine similarity
│
├── agent/
│   ├── resource_finder.py        # курсы / docs / GitHub (LLM → Web → Static)
│   ├── web_searcher.py          # live web search (DuckDuckGo + GitHub API)
│   ├── career_coach.py           # GenAI roadmap
│   ├── interview_coach.py       # банк 100+ вопросов
│   └── live_coach.py            # chatbot: mentor + "Roast My Stack"
│
├── utils/
│   ├── resume_parser.py          # текст → профиль (LLM / Regex)
│   ├── linkedin_parser.py       # LinkedIn PDF/text → профиль
│   ├── notebooklm_bridge.py     # NotebookLM MCP мост (Platform Layer)
│   ├── job_parser.py             # вакансия → навыки + gap
│   ├── pdf_parser.py            # PDF → текст → профиль
│   └── llm_client.py            # универсальный LLM клиент
│
├── telegram_bot/
│   ├── bot.py                   # Telegram бот (/jobs, /predict, PDF)
│   ├── job_scraper.py           # парсер IT-каналов + mock
│   └── auth.py                  # авторизация Telegram
│
├── app/
│   ├── app.py                   # Streamlit UI (8 вкладок)
│   └── dashboard.py             # Balance Wheel, RPG Tree, Wrapped
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
| 1 | **CV Upload** | Загрузить PDF резюме | Авто-извлечение навыков |
| 2 | **CV Upload** | LinkedIn Import → вставить текст | Парсинг LinkedIn профиля |
| 3 | **CV Upload** | NotebookLM → вставить summary | Извлечение через MCP / текст |
| 4 | **Predict** | Открыть вкладку | KPI + ⚖️ Balance Wheel + Top-3 ML + JSON |
| 5 | **Predict** | Выбрать целевую роль | 🎮 RPG Tech Tree (locked/unlocked) |
| 6 | **Predict** | Generate Semester Wrapped | 🎁 LinkedIn-ready PNG карточка |
| 7 | **CV Roast** | Slider + Run Roast | 🔥 Жёсткий разбор резюме (rule / LLM) |
| 8 | **Live Coach** | Написать вопрос | 💬 Mentor-режим: советы по карьере |
| 9 | **Live Coach** | Toggle "Roast My Stack" | 🔥 Агрессивный tech-lead roast |
| 10 | **Roadmap** | Выбрать роль → Generate | Поэтапный AI-план |
| 11 | **Interview** | Выбрать роль → Вопросы | 10+ вопросов по пробелам |
| 12 | **Telegram Jobs** | Кнопка Найти | Топ вакансий из каналов |
| 13 | *(бонус)* | Ввести API-ключ | LLM vs fallback живьём |

---

## 🗺️ Roadmap проекта

### ✅ Реализовано
- `predictor.py` — overlap score + нормализация + 11 алиасов
- `semantic_matcher.py` — TF-IDF cosine + hybrid score
- `resource_finder.py` — 20+ навыков, 4 типа ресурсов (LLM → Web → Static)
- `web_searcher.py` — live web search (DuckDuckGo + GitHub API)
- `career_coach.py` — roadmap по фазам с quick wins (LLM + template)
- `interview_coach.py` — 100+ вопросов по 15 навыкам (LLM + static)
- `live_coach.py` — chatbot с двумя персонами: mentor + "Roast My Stack"
- `resume_parser.py` — парсинг текста резюме (LLM + regex)
- `linkedin_parser.py` — парсинг LinkedIn PDF / текста (LLM + regex)
- `notebooklm_bridge.py` — MCP мост к NotebookLM (Platform Layer)
- `job_parser.py` — парсинг вакансии + gap analysis (LLM + regex)
- `pdf_parser.py` — PDF резюме → профиль (pdfplumber / pypdf / pdfminer)
- `job_scraper.py` — парсинг Telegram IT-каналов + mock (8 вакансий)
- `bot.py` — Telegram бот (/jobs, /predict, PDF upload)
- `app.py` — Streamlit UI: 8 вкладок, bar chart, radar chart
- `jobs.csv` — 21 роль × 8 навыков (расширенный датасет)
- Экспорт Semester Wrapped в PNG
- 29 unit-тестов

### 🔜 Следующие шаги
- [ ] Векторная БД (ChromaDB) для семантического поиска
- [ ] История сессий / прогресс-трекер
- [ ] Деплой на Streamlit Community Cloud

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
