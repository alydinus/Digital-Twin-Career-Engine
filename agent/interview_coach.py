"""
Agent Layer — Interview Coach (GenAI)

Генерирует вопросы для собеседования и примерные ответы на основе:
  - целевой роли
  - недостающих навыков (слабые места)
  - имеющихся навыков (сильные стороны)

Режимы:
  1. Static bank  — словарь из 100+ вопросов по навыкам
  2. LLM mode     — персонализированные вопросы через Claude/GPT
"""

from __future__ import annotations
from utils.llm_client import call_llm_json, is_available
import os
import random

# ---------------------------------------------------------------------------
# Статический банк вопросов по навыкам
# ---------------------------------------------------------------------------

QUESTION_BANK: dict[str, list[dict]] = {
    "python": [
        {"q": "Объясни разницу между list и tuple. Когда что использовать?",
         "hint": "Mutability, performance, use cases"},
        {"q": "Что такое GIL и как он влияет на многопоточность?",
         "hint": "Global Interpreter Lock, threading vs multiprocessing"},
        {"q": "Как работают генераторы и в чём их преимущество перед списками?",
         "hint": "yield, lazy evaluation, memory"},
        {"q": "Объясни декораторы на примере. Напиши свой декоратор.",
         "hint": "functools.wraps, closure, @syntax"},
    ],
    "sql": [
        {"q": "Чем отличается INNER JOIN от LEFT JOIN? Приведи пример.",
         "hint": "NULL rows, result set size"},
        {"q": "Что такое индекс в БД? Когда он замедляет работу?",
         "hint": "B-tree, write overhead, cardinality"},
        {"q": "Объясни разницу между WHERE и HAVING.",
         "hint": "Execution order, aggregation"},
        {"q": "Как работает транзакция? Что такое ACID?",
         "hint": "Atomicity, Consistency, Isolation, Durability"},
    ],
    "docker": [
        {"q": "Чем контейнер отличается от виртуальной машины?",
         "hint": "OS kernel sharing, overhead, startup time"},
        {"q": "Объясни разницу между COPY и ADD в Dockerfile.",
         "hint": "URL support, tar extraction"},
        {"q": "Что такое multi-stage build и зачем он нужен?",
         "hint": "Image size, build dependencies"},
        {"q": "Как организовать взаимодействие нескольких контейнеров?",
         "hint": "docker-compose, networks, volumes"},
    ],
    "kubernetes": [
        {"q": "Объясни разницу между Pod, Deployment и Service.",
         "hint": "Abstraction layers, scaling, routing"},
        {"q": "Что такое liveness probe и readiness probe?",
         "hint": "Health checks, traffic routing"},
        {"q": "Как работает горизонтальное масштабирование в K8s?",
         "hint": "HPA, metrics, replicas"},
        {"q": "Что такое ConfigMap и Secret? Чем они отличаются?",
         "hint": "Environment config vs sensitive data"},
    ],
    "aws": [
        {"q": "Объясни разницу между EC2, ECS и Lambda.",
         "hint": "VMs vs containers vs serverless"},
        {"q": "Что такое S3 versioning и когда его включать?",
         "hint": "Accidental deletion, compliance"},
        {"q": "Как работает IAM? Что такое принцип наименьших привилегий?",
         "hint": "Roles, policies, trust"},
        {"q": "Чем RDS отличается от DynamoDB? Когда что выбрать?",
         "hint": "Relational vs NoSQL, scaling patterns"},
    ],
    "system design": [
        {"q": "Как бы ты спроектировал URL shortener типа bit.ly?",
         "hint": "Hashing, DB choice, caching, scalability"},
        {"q": "Объясни CAP теорему на конкретном примере.",
         "hint": "Consistency, Availability, Partition tolerance"},
        {"q": "Как масштабировать систему от 1000 до 1 млн пользователей?",
         "hint": "Load balancer, caching, DB sharding, CDN"},
        {"q": "Что такое rate limiting и как его реализовать?",
         "hint": "Token bucket, sliding window, Redis"},
    ],
    "machine learning": [
        {"q": "Объясни разницу между overfitting и underfitting. Как бороться?",
         "hint": "Regularization, dropout, more data, cross-validation"},
        {"q": "Что такое gradient descent? Объясни learning rate.",
         "hint": "Optimization, local minima, batch/mini-batch/SGD"},
        {"q": "Чем отличается precision от recall? Когда что важнее?",
         "hint": "False positives vs false negatives, F1, use cases"},
        {"q": "Как работает Random Forest? Почему он лучше одного дерева?",
         "hint": "Ensemble, bagging, feature importance"},
    ],
    "pytorch": [
        {"q": "Объясни разницу между .detach() и .no_grad().",
         "hint": "Gradient computation, memory, inference"},
        {"q": "Что такое DataLoader и зачем он нужен?",
         "hint": "Batching, shuffling, num_workers"},
        {"q": "Как реализовать кастомную функцию потерь?",
         "hint": "nn.Module, forward pass, differentiability"},
        {"q": "Объясни цикл обучения: forward → loss → backward → step.",
         "hint": "optimizer.zero_grad(), loss.backward(), optimizer.step()"},
    ],
    "react": [
        {"q": "Объясни разницу между useState и useReducer.",
         "hint": "Complex state, action dispatch, redux-like"},
        {"q": "Что такое Virtual DOM и как React использует reconciliation?",
         "hint": "Diffing algorithm, re-renders, keys"},
        {"q": "Когда использовать useCallback и useMemo?",
         "hint": "Referential equality, expensive computations, deps"},
        {"q": "Объясни концепцию lifting state up.",
         "hint": "Shared state, parent-child communication"},
    ],
    "ci/cd": [
        {"q": "Чем отличается Continuous Integration от Continuous Delivery?",
         "hint": "Automation level, deployment frequency, risk"},
        {"q": "Как организовать безопасное хранение секретов в CI/CD?",
         "hint": "Secrets manager, env vars, vault"},
        {"q": "Что такое blue-green deployment?",
         "hint": "Zero downtime, rollback, traffic switching"},
        {"q": "Как реализовать rollback при неудачном деплое?",
         "hint": "Feature flags, canary, previous image"},
    ],
    "microservices": [
        {"q": "Какие паттерны коммуникации между микросервисами ты знаешь?",
         "hint": "Sync REST/gRPC, async Kafka/RabbitMQ, pros/cons"},
        {"q": "Что такое saga pattern и зачем он нужен?",
         "hint": "Distributed transactions, compensating transactions"},
        {"q": "Как обеспечить observability в микросервисной архитектуре?",
         "hint": "Logging, tracing, metrics, correlation ID"},
        {"q": "Что такое circuit breaker и когда его применять?",
         "hint": "Resilience, cascade failures, Hystrix/Resilience4j"},
    ],
    "rest api": [
        {"q": "Объясни принципы REST. Что делает API RESTful?",
         "hint": "Stateless, uniform interface, HATEOAS"},
        {"q": "Чем отличается PUT от PATCH?",
         "hint": "Full vs partial update, idempotency"},
        {"q": "Как версионировать API? Плюсы и минусы подходов.",
         "hint": "URL path, header, query param"},
        {"q": "Что такое idempotency и как её обеспечить?",
         "hint": "HTTP methods, retry safety, unique keys"},
    ],
    "postgresql": [
        {"q": "Что такое VACUUM и ANALYZE? Когда запускать?",
         "hint": "Dead tuples, autovacuum, statistics"},
        {"q": "Объясни разницу между sequence scan и index scan.",
         "hint": "EXPLAIN ANALYZE, selectivity, cost"},
        {"q": "Как работает JSONB в PostgreSQL? Когда его использовать?",
         "hint": "Binary storage, GIN indexes, vs JSON"},
        {"q": "Что такое connection pooling? Зачем pgBouncer?",
         "hint": "Connection overhead, max_connections, modes"},
    ],
    "terraform": [
        {"q": "Объясни разницу между terraform plan и terraform apply.",
         "hint": "Dry run, state diff, safety"},
        {"q": "Что такое terraform state и почему его нельзя удалять?",
         "hint": "Resource tracking, remote state, locking"},
        {"q": "Как организовать переиспользуемую инфраструктуру через модули?",
         "hint": "inputs/outputs, versioning, DRY"},
        {"q": "Как управлять несколькими окружениями (dev/stage/prod)?",
         "hint": "workspaces, separate state, tfvars"},
    ],
    "mlops": [
        {"q": "Что такое model drift и как его отслеживать?",
         "hint": "Data drift, concept drift, monitoring metrics"},
        {"q": "Объясни процесс CI/CD для ML модели.",
         "hint": "Training pipeline, validation, registry, deployment"},
        {"q": "Что такое feature store и зачем он нужен?",
         "hint": "Feature reuse, consistency, online/offline"},
        {"q": "Как версионировать ML модели и датасеты?",
         "hint": "MLflow, DVC, model registry"},
    ],
}

# Общие вопросы для поведенческого интервью
BEHAVIORAL_QUESTIONS = [
    {"q": "Расскажи о сложном техническом решении, которое тебе пришлось принять.",
     "hint": "STAR метод: Situation, Task, Action, Result"},
    {"q": "Как ты расставляешь приоритеты, когда есть несколько срочных задач?",
     "hint": "Матрица приоритетов, коммуникация с командой"},
    {"q": "Приведи пример, когда ты допустил ошибку. Что сделал?",
     "hint": "Честность, уроки, системные изменения"},
    {"q": "Как ты учишь новые технологии? Последнее, что изучил?",
     "hint": "Конкретный пример, методология"},
]


# ---------------------------------------------------------------------------
# Генерация вопросов
# ---------------------------------------------------------------------------

def _static_questions(
    missing_skills: list[str],
    matched_skills: list[str],
    n_per_skill: int = 2,
) -> dict:
    """Подбирает вопросы из статического банка."""
    questions = {"missing": [], "strengths": [], "behavioral": []}

    # Вопросы по слабым местам
    for skill in missing_skills:
        bank = QUESTION_BANK.get(skill.lower(), [])
        selected = random.sample(bank, min(n_per_skill, len(bank))) if bank else []
        for item in selected:
            questions["missing"].append({
                "skill": skill,
                "question": item["q"],
                "hint": item["hint"],
                "type": "technical",
            })

    # Вопросы по сильным сторонам (уточняющие)
    for skill in matched_skills[:3]:
        bank = QUESTION_BANK.get(skill.lower(), [])
        if bank:
            item = random.choice(bank)
            questions["strengths"].append({
                "skill": skill,
                "question": item["q"],
                "hint": item["hint"],
                "type": "technical",
            })

    # Поведенческие вопросы
    questions["behavioral"] = random.sample(BEHAVIORAL_QUESTIONS, min(3, len(BEHAVIORAL_QUESTIONS)))

    return questions


def _llm_questions(
    target_role: str,
    missing_skills: list[str],
    matched_skills: list[str],
) -> dict:
    """Генерирует персонализированные вопросы через LLM."""
    if not is_available():
        raise ValueError("LLM_API_KEY не задан")

    prompt = f"""
Ты — опытный технический интервьюер для позиции {target_role}.

Профиль кандидата:
- Знает: {', '.join(matched_skills)}
- Пробелы: {', '.join(missing_skills)}

Сгенерируй персонализированный список вопросов для собеседования.
Верни ТОЛЬКО валидный JSON:
{{
  "missing": [
    {{"skill": "название навыка", "question": "вопрос", "hint": "подсказка для ответа", "type": "technical"}}
  ],
  "strengths": [
    {{"skill": "название навыка", "question": "углублённый вопрос", "hint": "подсказка", "type": "technical"}}
  ],
  "behavioral": [
    {{"question": "поведенческий вопрос", "hint": "что хочет услышать интервьюер"}}
  ]
}}

Вопросы должны быть конкретными, реалистичными и соответствовать уровню кандидата.
""".strip()

    return call_llm_json(prompt, max_tokens=2000)


def generate_interview_questions(
    target_role: str,
    missing_skills: list[str],
    matched_skills: list[str],
    n_per_skill: int = 2,
    use_llm: bool = False,
) -> dict:
    """
    Генерирует вопросы для подготовки к собеседованию.

    Returns:
        {
          "target_role": str,
          "source":      "static" | "llm",
          "missing":     [{skill, question, hint, type}],   # вопросы по пробелам
          "strengths":   [{skill, question, hint, type}],   # вопросы по сильным сторонам
          "behavioral":  [{question, hint}],                 # поведенческие вопросы
        }
    """
    if use_llm or os.getenv("LLM_API_KEY"):
        try:
            result = _llm_questions(target_role, missing_skills, matched_skills)
            result["target_role"] = target_role
            result["source"] = "llm"
            return result
        except Exception as e:
            print(f"[InterviewCoach] LLM недоступен ({e}), использую банк вопросов.")

    result = _static_questions(missing_skills, matched_skills, n_per_skill)
    result["target_role"] = target_role
    result["source"] = "static"
    return result


# ---------------------------------------------------------------------------
# CLI тест
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, ".")
    from model.predictor import load_profile, load_jobs, predict_top_roles

    profile = load_profile("data/profile.json")
    jobs    = load_jobs("data/jobs.csv")
    top     = predict_top_roles(profile, jobs, top_n=1)
    best    = top[0]

    qs = generate_interview_questions(
        target_role    = best["role"],
        missing_skills = best["missing_skills"],
        matched_skills = best["matched_skills"],
    )

    print(f"\n🎤 Вопросы для: {qs['target_role']} [{qs['source']}]\n")
    print("--- Слабые места ---")
    for q in qs["missing"][:3]:
        print(f"  [{q['skill']}] {q['question']}")
        print(f"  Hint: {q['hint']}\n")
    print("--- Поведенческие ---")
    for q in qs["behavioral"]:
        print(f"  {q['question']}")
