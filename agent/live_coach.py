"""
Agent Layer — Live Coach Chatbot

Multi-turn chat with two personas:
  - "mentor"  : helpful, structured career mentor (default)
  - "roast"   : aggressive tech-lead persona that humorously critiques
                YouTube procrastination, messy code, and skill gaps.

Required by the Final Generative AI Project PDF:
  > The Live Coach Chatbot: A chat interface to talk directly to your
  > Digital Twin. You must include a "Roast My Stack" toggle button that
  > changes the AI's persona into an aggressive tech lead who humorously
  > critiques your YouTube procrastination and messy code.

Public API:
    coach_reply(messages, persona, profile, predictions=None) -> str

`messages` is a list[{"role": "user"|"assistant", "content": str}] with
the latest user turn at the end.

Falls back to rule-based responses when LLM_API_KEY is not configured.
"""

from __future__ import annotations

import os
import random
from typing import Any

# ---------------------------------------------------------------------------
# System prompts per persona
# ---------------------------------------------------------------------------

_MENTOR_SYSTEM = """\
You are the user's Digital Twin Career Mentor — a calm, structured,
empathetic senior engineer. You give concrete, actionable advice grounded
in the user's profile and ML predictions. Avoid hype words, prefer
specifics: name technologies, mention concrete next steps, suggest
1-2 quality resources. Reply in the same language the user wrote in
(usually Russian or English). Keep replies to 4-8 sentences max unless
the user explicitly asks for a long answer.
"""

_ROAST_SYSTEM = """\
You are "Roast My Stack" — an aggressive but funny senior tech lead who
has just opened the user's CV and YouTube history and is appalled. Speak
sharply, use playful sarcasm, brutal-honest critique. Mock procrastination
patterns ("you watched 14 hours of '10x developer' videos and still
can't write a docker-compose"), messy stacks ("five JS frameworks, zero
tests"), and gaps ("no Kubernetes in 2026? bold of you"). Keep it
funny, not cruel — punch up the criticism with wit. Always end with one
concrete "fine, here's what to actually do" sentence so the roast lands
useful. Reply in the user's language. 4-7 sentences.
"""


def _build_context_block(profile: dict, predictions: list[dict] | None) -> str:
    hard       = profile.get("hard_skills", [])
    soft       = profile.get("soft_skills", [])
    interests  = profile.get("interests", [])
    name       = profile.get("name", "—")

    lines = [
        f"USER PROFILE",
        f"  name:       {name}",
        f"  hard_skills ({len(hard)}): {', '.join(hard) if hard else '(none)'}",
        f"  soft_skills ({len(soft)}): {', '.join(soft) if soft else '(none)'}",
        f"  interests:  {', '.join(interests) if interests else '(none)'}",
    ]
    if predictions:
        lines.append("ML PREDICTIONS (top-3):")
        for r in predictions[:3]:
            miss = ", ".join(r.get("missing_skills", [])[:5])
            lines.append(
                f"  - {r['role']}  fit={r['score_pct']}%  "
                f"missing=[{miss}]"
            )
    return "\n".join(lines)


def _to_dialog(messages: list[dict]) -> str:
    """Render history as plain dialog text for non-chat LLM endpoints."""
    out = []
    for m in messages:
        role = m["role"].upper()
        out.append(f"{role}: {m['content']}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

_MENTOR_RULES = [
    ("привет", "Привет. Что хочешь обсудить — навыки, карьерный путь, проект для портфолио?"),
    ("hello", "Hi. What do you want to talk about — skills, career path, a portfolio project?"),
    ("навык", "Глянь раздел Predict — там твои топ-3 ролей с дельтой по навыкам. По какому конкретно скиллу нужен план?"),
    ("skill",  "Open the Predict tab — your top-3 roles with skill deltas are there. Which specific skill do you want a plan for?"),
    ("совет",  "Конкретный совет: возьми один missing skill из top-1 роли, выдели 2 недели, сделай pet-проект и положи в GitHub. Какой скилл первый?"),
    ("план",   "Открой вкладку Roadmap — там сгенерируется фазовый план. Если хочешь, могу разобрать его по этапам."),
    ("резюме", "Закинь PDF в CV Upload, потом запусти CV Roast с harshness=8 — увидишь все красные флаги."),
]

_ROAST_RULES = [
    ("привет", "Привет. У тебя в стеке 3 языка и 0 тестов — мы точно идём здороваться, или работать? 😏"),
    ("hello",  "Hi. Three languages, zero tests in your repo. We chatting or shipping? 😏"),
    ("навык",  "Скилл? Ты последние 6 месяцев смотрел Fireship, а руками потрогал Kubernetes? Открой docs.k8s.io и не закрывай неделю."),
    ("skill",  "Skill? You binged Fireship for 6 months and never touched Kubernetes. Open docs.k8s.io and don't close it for a week."),
    ("совет",  "Совет: меньше тиктоков «топ-10 фреймворков», больше pull request'ов в один проект на 3 месяца. Один. Глубокий. Иди."),
    ("план",   "План? Сначала закрой 5 вкладок с туториалами. Потом возьми ОДИН missing skill и сделай ОДИН проект. Хватит коллекционировать звёздочки в GitHub."),
    ("резюме", "Резюме на 1 страницу, навыков на 2 страницы, реальных проектов — 0. Знакомо? Удали половину buzzwords и впиши то, что реально деплоил."),
]


def _rule_based_reply(persona: str, last_user_msg: str) -> str:
    rules = _ROAST_RULES if persona == "roast" else _MENTOR_RULES
    msg = (last_user_msg or "").lower()
    for trigger, reply in rules:
        if trigger in msg:
            return reply

    if persona == "roast":
        return random.choice([
            "Слушай, я бы ответил, но я ещё в шоке от твоего стека. Уточни вопрос — помогу.",
            "Эээ, переформулируй. Ты же не задаёшь такие же расплывчатые вопросы тимлиду на 1-on-1?",
            "Бро, конкретнее. «Что мне делать?» — это вопрос экзистенциальный. Назови скилл, я разнесу.",
        ])
    return random.choice([
        "Уточни вопрос — какой скилл/роль/этап карьеры тебя интересует?",
        "Дай контекст: цель на 3 месяца? на год? и я предложу конкретные шаги.",
        "Нужен план обучения, разбор резюме или подготовка к интервью? Выбери одно.",
    ])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def coach_reply(
    messages: list[dict],
    persona: str = "mentor",
    profile: dict | None = None,
    predictions: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Generate the next assistant turn.

    Returns:
        {"reply": str, "source": "llm" | "rule-based"}
    """
    persona = "roast" if persona == "roast" else "mentor"
    profile = profile or {}

    if not messages:
        return {"reply": "(empty conversation)", "source": "rule-based"}

    last_user = next((m["content"] for m in reversed(messages)
                       if m["role"] == "user"), "")

    api_key = os.getenv("LLM_API_KEY", "").strip()
    if api_key:
        try:
            from utils.llm_client import call_llm

            system    = _ROAST_SYSTEM if persona == "roast" else _MENTOR_SYSTEM
            ctx       = _build_context_block(profile, predictions)
            dialog    = _to_dialog(messages[-12:])  # cap history to 12 turns

            prompt = (
                f"{system}\n\n"
                f"--- USER CONTEXT ---\n{ctx}\n\n"
                f"--- CONVERSATION ---\n{dialog}\n\n"
                f"ASSISTANT:"
            )
            text = call_llm(prompt, max_tokens=600).strip()
            # strip leading "ASSISTANT:" if model echoes it
            if text.upper().startswith("ASSISTANT:"):
                text = text.split(":", 1)[1].strip()
            return {"reply": text, "source": "llm"}
        except Exception:
            pass

    return {"reply": _rule_based_reply(persona, last_user),
            "source": "rule-based"}


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    profile = {
        "name":        "Test User",
        "hard_skills": ["Python", "Docker", "Git", "SQL"],
        "soft_skills": ["communication"],
        "interests":   ["backend", "ML"],
    }
    preds = [
        {"role": "Backend Engineer", "score_pct": 50,
         "matched_skills": ["Python", "Docker"],
         "missing_skills": ["Kubernetes", "System Design"]},
    ]
    history = [{"role": "user", "content": "Какой план на ближайший месяц?"}]
    print("MENTOR:", coach_reply(history, "mentor", profile, preds))
    print("ROAST :", coach_reply(history, "roast",  profile, preds))
