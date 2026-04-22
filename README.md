# 🎯 Digital Twin Career Engine

Система анализа навыков и предсказания оптимальных карьерных направлений.
На входе — профиль пользователя; на выходе — топ профессий, недостающие навыки и ресурсы для их изучения.

---

## 📐 Архитектура проекта (5 слоёв)

| Слой | Назначение | Технологии |
|------|-----------|------------|
| **1. Platform Layer** | Источник данных (profile.json, jobs.csv) | JSON, CSV |
| **2. Model Layer** | ML-логика сопоставления навыков | Python, scikit-learn, pandas |
| **3. Agent Layer** | Автоматический поиск ресурсов | LLM / Web API |
| **4. Application Layer** | Интерфейс пользователя | Streamlit |
| **5. Infrastructure Layer** | Среда запуска | Python venv, локально |

---

## 🗂️ Структура проекта

```
genAi/
├── data/
│   ├── profile.json          # профиль пользователя (вход)
│   └── jobs.csv              # база профессий и требуемых навыков
├── model/
│   ├── __init__.py
│   └── predictor.py          # ML-матчинг skills ↔ roles
├── agent/
│   ├── __init__.py
│   └── resource_finder.py    # агент поиска курсов / доков
├── app/
│   └── app.py                # Streamlit UI
├── tests/
│   └── test_predictor.py     # unit-тесты модели
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Поэтапный план реализации

### ✅ Этап 0. Подготовка окружения
- Создать структуру папок (`data/`, `model/`, `agent/`, `app/`, `tests/`).
- Создать виртуальное окружение Python (`python -m venv venv`).
- Создать `requirements.txt` с зависимостями: `pandas`, `scikit-learn`, `streamlit`, `requests`, `python-dotenv`.
- Инициализировать `.gitignore`.

**Что нужно от пользователя:** ничего. Этап полностью автоматизируемый.

---

### 📥 Этап 1. Platform Layer — данные

**1.1. `data/profile.json`** — профиль пользователя.
```json
{
  "name": "Alydin",
  "hard_skills": ["Python", "SQL", "Docker", "Git"],
  "soft_skills": ["problem solving", "teamwork"],
  "interests": ["backend", "data", "AI"]
}
```

**1.2. `data/jobs.csv`** — база профессий.
```csv
role,required_skills
Backend Engineer,"Python;SQL;Docker;System Design;Microservices;Git"
DevOps Engineer,"Docker;Kubernetes;AWS;Linux;CI/CD;Bash"
Data Scientist,"Python;Pandas;SQL;ML;Statistics;NumPy"
ML Engineer,"Python;PyTorch;MLOps;Docker;SQL;Math"
Frontend Engineer,"JavaScript;React;HTML;CSS;TypeScript;Git"
```

**Что нужно от пользователя:**
- 👤 **Персональный профиль**: список hard skills, soft skills, интересов (или разрешение использовать шаблон).
- 📊 **База профессий**: либо подтверждение использовать базовый набор (5–10 ролей), либо кастомный список ролей.

---

### 🧠 Этап 2. Model Layer — ML-матчинг

Файл `model/predictor.py`:

- **`load_profile(path)`** — загрузка профиля.
- **`load_jobs(path)`** — загрузка базы ролей.
- **`compute_match_score(user_skills, role_skills)`** — Jaccard / overlap score:
  `score = |user ∩ role| / |role|`.
- **`find_missing_skills(user_skills, role_skills)`** — разница множеств.
- **`predict_top_roles(profile, jobs, top_n=3)`** — возвращает топ-N ролей со score и missing skills.
- (опционально) **TF-IDF / embedding similarity** для «похожих» навыков (например, `pytorch` ≈ `torch`).

**Тест:** `tests/test_predictor.py` — на фиктивном профиле должен выдать предсказуемый ранжированный список.

**Что нужно от пользователя:** ничего, если профиль + jobs.csv уже готовы.

---

### 🤖 Этап 3. Agent Layer — поиск ресурсов

Файл `agent/resource_finder.py`:

- **`find_resources(skill)`** — принимает навык (например, `"Kubernetes"`), возвращает:
  - официальную документацию,
  - 1–2 GitHub-репозитория (через GitHub Search API),
  - 1–2 курса (статический маппинг или LLM-запрос).
- **Fallback-режим:** если нет LLM-ключа — статический словарь `skill → [ссылки]`.
- **LLM-режим (опционально):** обёртка над Anthropic API / OpenAI для генерации 3 наиболее релевантных ссылок.

**Что нужно от пользователя:**
- 🔑 **API-ключ** (опционально): Anthropic / OpenAI / GitHub token. Если не предоставлен — работает fallback-режим.

---

### 🖥️ Этап 4. Application Layer — Streamlit UI

Файл `app/app.py`:

Секции интерфейса:
1. **Профиль** — загрузка или редактирование `profile.json`.
2. **Кнопка «Predict»** — запускает `predictor.predict_top_roles`.
3. **Результаты** — топ-3 роли, score (progress bar), missing skills (чипы/теги).
4. **Ресурсы** — при клике на missing skill агент подтягивает ссылки.
5. **Экспорт** — возможность сохранить отчёт в Markdown.

Запуск: `streamlit run app/app.py`.

**Что нужно от пользователя:** ничего.

---

### 🧪 Этап 5. Тестирование и валидация
- Unit-тесты: `pytest tests/`.
- Ручная проверка на 2–3 разных профилях (junior / middle / switcher).
- Sanity-check: ручной подсчёт score для одной роли и сравнение с выводом модели.

---

### 📦 Этап 6. Инфраструктура и упаковка
- `requirements.txt` зафиксировать (`pip freeze`).
- README дополнить инструкцией по запуску.
- (опционально) Dockerfile для воспроизводимого запуска.
- (опционально) деплой на Streamlit Community Cloud.

---

## 🎬 Демо-сценарий (защита проекта)
1. Открыть Streamlit-приложение.
2. Показать профиль пользователя.
3. Нажать **Predict** → показать топ-3 роли со score.
4. Раскрыть missing skills для каждой роли.
5. Кликнуть на skill → агент показывает ресурсы.
6. (бонус) Изменить профиль и переcчитать вживую.

---

## 📋 Сводка: что нужно от пользователя

| № | Что | Обязательно? | Как предоставить |
|---|-----|--------------|------------------|
| 1 | Персональный профиль (skills, interests) | ✅ Да | Отредактировать `profile.json` или продиктовать |
| 2 | Список целевых профессий | ⚠️ Желательно | Подтвердить дефолтный набор или прислать свой |
| 3 | API-ключ (Anthropic / OpenAI) | ❌ Нет | Положить в `.env` (`LLM_API_KEY=...`) |
| 4 | GitHub token | ❌ Нет | `.env` (`GITHUB_TOKEN=...`) |

Если пунктов 3–4 нет — агент работает в fallback-режиме со статическим словарём ресурсов.

---

## ⏱️ Примерный тайминг
- Этап 0–1: **~20 мин** (скелет + данные)
- Этап 2: **~40 мин** (predictor + тесты)
- Этап 3: **~30 мин** (resource finder)
- Этап 4: **~45 мин** (Streamlit UI)
- Этап 5–6: **~20 мин** (тесты, упаковка)

**Итого:** ~2.5 часа до рабочего MVP.

---

## ❓ Открытые вопросы перед стартом
1. Использовать **готовый шаблон профиля** или вы пришлёте свой?
2. Какие роли должны быть в `jobs.csv` — базовые 5 или расширенный список?
3. Нужен ли **LLM-агент** (с API-ключом) или достаточно статического fallback?
4. Нужна ли визуализация score (bar chart / radar chart)?
