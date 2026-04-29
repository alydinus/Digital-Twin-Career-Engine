# 🎯 Digital Twin Career Engine

> **Final Generative AI Project** — система, которая анализирует навыки,
> предсказывает карьерные направления, генерирует персональный план развития
> и готовит к собеседованию — с помощью GenAI.

---

## 🤖 Почему это GenAI-проект?

Семь генеративных компонентов, каждый работает в двух режимах:

| Компонент | Fallback (без ключа) | GenAI (с LLM_API_KEY) |
|---|---|---|
| **Resume / LinkedIn Parser** | Regex по словарю 60+ навыков | LLM извлекает навыки из произвольного текста |
| **Transcript Parser** | Rule-based course-to-skill mapping | Обогащает профиль академическими сигналами |
| **Job Parser** | Regex + паттерны секций вакансии | LLM структурирует JD → JSON |
| **Skill Matcher** | Exact overlap score + алиасы | Hybrid: overlap + TF-IDF cosine similarity |
| **Classical ML Recommender** | Bundled dataset | KNN over uploaded jobs CSV |
| **Career Coach** | Шаблонный roadmap по сложности | LLM генерирует персональный план |
| **Interview Coach** | Банк 100+ вопросов по навыкам | LLM создаёт персональные вопросы |
| **Live Coach** | Rule-based mentor / roast replies | LLM-чат с контекстом из ML-предсказания |
| **Resource Finder** | Статический словарь 20+ навыков | Live web search + HTML parsing top-3 results / LLM |

---

## 🏗️ Архитектура (5 слоёв)

```
 Input: profile / resume text / job description
            │
            ▼
 ┌──────────────────────────────────────────────────────┐
 │  1. PLATFORM LAYER                                   │
 │     CV + LinkedIn + Transcript ─ merged profile      │
 │     data/profile.json  ── шаблон профиля             │
 │     data/jobs.csv      ── 5 ролей × 8 навыков        │
 │     utils/resume_parser.py  ◄── LLM / Regex          │
 │     utils/job_parser.py     ◄── LLM / Regex          │
 │     utils/profile_ingestion.py ◄── source merging    │
 │     utils/notebooklm_bridge.py ◄── MCP config        │
 └─────────────────────┬────────────────────────────────┘
                       │  структурированный профиль
                       ▼
 ┌──────────────────────────────────────────────────────┐
 │  2. MODEL LAYER                                      │
 │     model/predictor.py        ── hybrid scoring      │
 │     model/semantic_matcher.py ── TF-IDF cosine       │
 │     model/job_dataset_trainer.py ─ KNN recommender   │
 └─────────────────────┬────────────────────────────────┘
                       │  top-N roles + scores + missing
                       ▼
 ┌──────────────────────────────────────────────────────┐
 │  3. AGENT LAYER  (GenAI core)                        │
 │     agent/resource_finder.py ─ live web search/HTML  │
 │     agent/career_coach.py    ── roadmap генератор    │
 │     agent/interview_coach.py ── вопросы к интервью   │
 │     agent/live_coach.py      ── mentor/roast chat    │
 └─────────────────────┬────────────────────────────────┘
                       │  ресурсы + план + вопросы
                       ▼
 ┌──────────────────────────────────────────────────────┐
 │  4. APPLICATION LAYER — Streamlit UI (8 вкладок)     │
 │     📄 CV Upload · 🎯 Predict · 🔥 CV Roast          │
 │     📨 Telegram Jobs · 🗺️ Roadmap · 🤖 Live Coach    │
 │     🎤 Interview · ℹ️ About                          │
 │                                                      │
 │  Виджеты во вкладке Predict (по требованию PDF):     │
 │     ⚖️ Balance Wheel  ── radar Hard vs Soft skills    │
 │     🎮 RPG Tech Tree  ── locked / unlocked nodes      │
 │     🎁 Semester Wrapped ─ shareable LinkedIn PNG      │
 │     🤖 Live Coach ─ отдельный chat tab + roast toggle │
 └──────────────────────────────────────────────────────┘
                       │
 ┌──────────────────────────────────────────────────────┐
 │  5. INFRASTRUCTURE LAYER                             │
 │     Python 3.10 · venv · локально / Streamlit Cloud  │
 │     .env ── LLM_API_KEY, LLM_PROVIDER                │
 └──────────────────────────────────────────────────────┘
```

---

## 🖥️ Infrastructure Layer — где что считается

PDF требует явно перечислить, что крутится локально на CPU, а что — в облаке.

| Компонент | Где работает | Нагрузка | Зависимости |
|---|---|---|---|
| Streamlit UI (`app/app.py`) | **Локально, CPU** | низкая | streamlit, matplotlib |
| Dashboard widgets (`app/dashboard.py`) | **Локально, CPU** | низкая (matplotlib) | matplotlib, numpy |
| Predictor — hybrid (`model/predictor.py`) | **Локально, CPU** | O(N×M) ≈ ms | pandas, scikit-learn |
| Semantic Matcher — TF-IDF (`model/semantic_matcher.py`) | **Локально, CPU** | O(N×M) ≈ ms | scikit-learn |
| Resume / PDF Parser (regex fallback) | **Локально, CPU** | низкая | pdfplumber / regex |
| CV Roast (rule-based) | **Локально, CPU** | мгновенно | — |
| Resource Finder (static dict) | **Локально, CPU** | мгновенно | — |
| Live Resource Agent (`agent/resource_finder.py`) | **Cloud / сеть** | HTTP + HTML parse | requests, beautifulsoup4 |
| Live Coach (`agent/live_coach.py`) | **Локально / Cloud** | низкая | rule-based / LLM API |
| Telegram Job Scraper (mock) | **Локально, CPU** | низкая | — |
| Telegram Job Scraper (Telethon) | **Локально + Telegram cloud** | сетевая | telethon, MTProto |
| LLM-режим Resume Parser | **Cloud (Anthropic / OpenAI)** | API-вызов | LLM_API_KEY |
| LLM-режим Career Coach | **Cloud (Anthropic / OpenAI)** | API-вызов | LLM_API_KEY |
| LLM-режим Interview Coach | **Cloud (Anthropic / OpenAI)** | API-вызов | LLM_API_KEY |
| LLM-режим Resource Finder | **Cloud (Anthropic / OpenAI)** | API-вызов | LLM_API_KEY |
| LLM-режим CV Roast | **Cloud (Anthropic / OpenAI)** | API-вызов | LLM_API_KEY |
| NotebookLM MCP Bridge (`utils/notebooklm_bridge.py`) | **Cloud (Google) + Antigravity MCP** | API-вызов | notebooklm-mcp-server |

**Hosting вариант:**

* Локально: `python3 -m venv venv && streamlit run app/app.py` — всё на твоём CPU, LLM-фичи опциональны.
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
│   ├── predictor.py          # hybrid score: overlap + TF-IDF semantic
│   ├── semantic_matcher.py   # TF-IDF cosine similarity + hybrid score
│   └── job_dataset_trainer.py # classical ML KNN over jobs CSV
│
├── agent/
│   ├── resource_finder.py    # static + live web search + HTML parsing
│   ├── career_coach.py       # поэтапный roadmap по фазам (LLM + template)
│   ├── interview_coach.py    # банк 100+ вопросов по навыкам (LLM + static)
│   └── live_coach.py         # чат-агент с Roast My Stack
│
├── utils/
│   ├── resume_parser.py      # текст резюме → структурированный профиль
│   ├── job_parser.py         # текст вакансии → навыки + gap analysis
│   ├── profile_ingestion.py  # LinkedIn + transcript → merged profile
│   └── notebooklm_bridge.py  # MCP config для NotebookLM + Antigravity
│
├── app/
│   ├── app.py                # Streamlit UI — 8 вкладок
│   └── dashboard.py          # Balance Wheel / RPG Tech Tree / Wrapped
│
├── tests/
│   ├── test_predictor.py
│   └── test_agent_layer.py
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
python3 -m venv venv && source venv/bin/activate
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
| 2 | **Predict** | Кнопка Predict | Radar chart + top-3 JSON + hybrid ML scores |
| 3 | **Predict** | Включить `Live web search` → выбрать missing skill | Агент парсит top-3 live results и показывает актуальные ресурсы |
| 4 | **Live Coach** | Включить `Roast My Stack` | Persona switch: mentor ↔ aggressive tech lead |
| 5 | **Roadmap** | Выбрать роль → Сгенерировать | Поэтапный план: фазы, сроки, quick wins |
| 6 | **Interview** | Выбрать роль → Вопросы | Тех. вопросы по пробелам + behavioral |
| 7 | **Telegram Jobs** | Найти вакансии | Match against CV + инфографика |
| 8 | **About** | Открыть MCP config | 5 слоёв + infra split + NotebookLM bridge |

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

**Тесты покрывают:** нормализацию · hybrid score · missing skills · MCP bridge · agent fallback.


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
│   ├── predictor.py              # hybrid score + алиасы
│   └── semantic_matcher.py      # TF-IDF cosine similarity
│
├── agent/
│   ├── resource_finder.py        # static + live web search + HTML parse
│   ├── career_coach.py           # GenAI roadmap
│   ├── interview_coach.py       # банк 100+ вопросов
│   └── live_coach.py            # mentor / roast chat
│
├── utils/
│   ├── resume_parser.py          # текст → профиль (LLM / Regex)
│   ├── job_parser.py             # вакансия → навыки + gap
│   ├── pdf_parser.py             # PDF → текст → профиль
│   └── notebooklm_bridge.py      # MCP config for NotebookLM
│
├── telegram_bot/
│   ├── bot.py                   # Telegram бот (/jobs, /predict, PDF)
│   └── job_scraper.py           # парсер IT-каналов + mock
│
├── app/
│   ├── app.py                    # Streamlit UI (8 вкладок)
│   └── dashboard.py              # charts + wrapped card
│
├── tests/
│   ├── test_predictor.py
│   └── test_agent_layer.py
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
| 1 | **CV Upload** | Загрузить Resume + LinkedIn + Transcript | Merged digital profile |
| 2 | **Predict** | Загрузить jobs CSV и выбрать engine | Hybrid matcher или classical ML KNN |
| 3 | **Predict** | Открыть вкладку | KPI + ⚖️ Balance Wheel + Top-3 ML + JSON |
| 4 | **Predict** | Выбрать целевую роль | 🎮 RPG Tech Tree (locked/unlocked) |
| 5 | **Predict** | Generate Semester Wrapped | 🎁 LinkedIn-ready PNG карточка |
| 6 | **CV Roast** | Slider + Run Roast | 🔥 Жёсткий разбор резюме (rule / LLM) |
| 7 | **Live Coach** | Включить `Roast My Stack` | Persona switch + multi-turn chat |
| 8 | **Roadmap** | Выбрать роль → Generate | Поэтапный AI-план |
| 9 | **Interview** | Выбрать роль → Вопросы | 10+ вопросов по пробелам |
| 10 | **Telegram Jobs** | Кнопка Найти | Топ вакансий из каналов |

---

## 🗺️ Roadmap проекта

### ✅ Реализовано
- `profile_ingestion.py` — LinkedIn + transcript → merged profile
- `predictor.py` — hybrid score: overlap + semantic TF-IDF
- `semantic_matcher.py` — TF-IDF cosine + hybrid score
- `job_dataset_trainer.py` — classical ML KNN over uploaded jobs CSV
- `resource_finder.py` — static + live web search + HTML parsing top-3
- `career_coach.py` — roadmap по фазам с quick wins (LLM + template)
- `interview_coach.py` — 100+ вопросов по 15 навыкам (LLM + static)
- `live_coach.py` — mentor / roast persona chat
- `resume_parser.py` — парсинг текста резюме (LLM + regex)
- `job_parser.py` — парсинг вакансии + gap analysis (LLM + regex)
- `pdf_parser.py` — PDF резюме → профиль (pdfplumber / pypdf / pdfminer)
- `notebooklm_bridge.py` — воспроизводимый MCP config для Antigravity
- `job_scraper.py` — парсинг Telegram IT-каналов + mock (8 вакансий)
- `bot.py` — Telegram бот (/jobs, /predict, PDF upload)
- `app.py` — Streamlit UI: 8 вкладок, chat, live resource search
- `dashboard.py` — Balance Wheel, RPG Tech Tree, Semester Wrapped
- тесты на predictor, agent layer и MCP bridge

### 🔜 Следующие шаги
- [ ] LinkedIn/HH.ru scraper
- [ ] Расширение `jobs.csv` до 20+ ролей
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
