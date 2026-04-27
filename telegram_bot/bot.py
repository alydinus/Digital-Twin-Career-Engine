"""
Telegram Bot — Digital Twin Career Engine

Позволяет использовать систему прямо в Telegram:
  /start          — приветствие и инструкция
  /jobs           — показать свежие вакансии по профилю
  /predict        — предсказать топ-3 роли
  /profile        — показать текущий профиль
  Файл PDF        — загрузить резюме и автоматически обновить профиль

Установка:
  pip install python-telegram-bot python-dotenv

Настройка (.env):
  TELEGRAM_BOT_TOKEN=your_bot_token   # получить у @BotFather
  LLM_API_KEY=...                     # опционально

Запуск:
  python telegram_bot/bot.py
"""

from __future__ import annotations
import os
import sys
import json
import asyncio
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_profile() -> dict:
    from model.predictor import load_profile
    return load_profile(ROOT / "data" / "profile.json")


def _format_jobs(jobs, max_n: int = 5) -> str:
    if not jobs:
        return "❌ Подходящих вакансий не найдено. Попробуй снизить порог совпадения."

    lines = [f"🎯 Топ-{min(max_n, len(jobs))} вакансий по твоему профилю:\n"]
    for i, j in enumerate(jobs[:max_n], 1):
        seniority_emoji = {"senior":"🔴","middle":"🟡","junior":"🟢"}.get(j.seniority,"🟡")
        lines.append(
            f"{i}. {seniority_emoji} *{j.role_name}*\n"
            f"   📊 Match: *{j.match_pct}%* | {j.channel_name}\n"
            f"   ✅ Есть: {', '.join(j.you_have[:4]) or '—'}\n"
            f"   ❌ Нужно: {', '.join(j.you_need[:4]) or '—'}\n"
            f"   🔗 [Открыть]({j.url})\n"
        )
    return "\n".join(lines)


def _format_predict(top_roles) -> str:
    medals = ["🥇", "🥈", "🥉"]
    lines  = ["🎯 *Топ профессий по твоему профилю:*\n"]
    for i, r in enumerate(top_roles):
        m = medals[i] if i < 3 else f"{i+1}."
        bar = "█" * (r["score_pct"] // 10) + "░" * (10 - r["score_pct"] // 10)
        lines.append(
            f"{m} *{r['role']}* — {r['score_pct']}%\n"
            f"   `{bar}`\n"
            f"   ✅ {', '.join(r['matched_skills']) or '—'}\n"
            f"   ❌ {', '.join(r['missing_skills'][:4]) or '—'}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bot handlers
# ---------------------------------------------------------------------------

async def cmd_start(update, context):
    text = (
        "👋 *Digital Twin Career Engine*\n\n"
        "Я анализирую твои навыки и нахожу подходящие вакансии в Telegram-каналах.\n\n"
        "📋 *Команды:*\n"
        "/profile — посмотреть текущий профиль\n"
        "/predict — топ-3 профессии по навыкам\n"
        "/jobs    — свежие вакансии по профилю\n"
        "/skills  — обновить навыки вручную\n\n"
        "📄 *Загрузи PDF резюме* — я автоматически обновлю профиль\n\n"
        "⚙️ Настрой .env для реального поиска по Telegram-каналам."
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_profile(update, context):
    try:
        profile = _load_profile()
        text = (
            f"👤 *Профиль: {profile.get('name','User')}*\n\n"
            f"🛠 *Hard Skills:*\n{', '.join(profile.get('hard_skills',[])) or '—'}\n\n"
            f"🤝 *Soft Skills:*\n{', '.join(profile.get('soft_skills',[])) or '—'}\n\n"
            f"💡 *Интересы:*\n{', '.join(profile.get('interests',[])) or '—'}"
        )
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


async def cmd_predict(update, context):
    await update.message.reply_text("🔄 Анализирую навыки...")
    try:
        from model.predictor import load_jobs, predict_top_roles
        profile   = _load_profile()
        jobs      = load_jobs(ROOT / "data" / "jobs.csv")
        top_roles = predict_top_roles(profile, jobs, top_n=3)
        text      = _format_predict(top_roles)
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка предсказания: {e}")


async def cmd_jobs(update, context):
    await update.message.reply_text("🔍 Ищу вакансии...")
    try:
        from telegram_bot.job_scraper import scrape_jobs
        profile = _load_profile()
        jobs    = scrape_jobs(profile, min_match_pct=25)
        text    = _format_jobs(jobs, max_n=5)
        await update.message.reply_text(text, parse_mode="Markdown",
                                        disable_web_page_preview=True)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка поиска: {e}")


async def cmd_skills(update, context):
    """Обновить hard skills через команду /skills Python, Docker, SQL"""
    args = context.args
    if not args:
        await update.message.reply_text(
            "📝 Укажи навыки через запятую:\n"
            "`/skills Python, Docker, SQL, Git`",
            parse_mode="Markdown"
        )
        return

    new_skills = [s.strip() for s in " ".join(args).split(",") if s.strip()]
    try:
        profile_path = ROOT / "data" / "profile.json"
        with open(profile_path, encoding="utf-8") as f:
            profile = json.load(f)

        profile["hard_skills"] = new_skills
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

        await update.message.reply_text(
            f"✅ Hard Skills обновлены:\n{', '.join(new_skills)}"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


async def handle_document(update, context):
    """Обрабатывает загруженные PDF-файлы."""
    doc = update.message.document
    if not doc or not doc.file_name.lower().endswith(".pdf"):
        await update.message.reply_text("📄 Загрузи PDF-файл резюме.")
        return

    await update.message.reply_text("📄 Получил резюме, обрабатываю...")

    try:
        from utils.pdf_parser import parse_pdf_resume

        # Скачиваем файл
        file = await context.bot.get_file(doc.file_id)
        pdf_bytes = await file.download_as_bytearray()

        # Парсим
        profile = parse_pdf_resume(bytes(pdf_bytes))

        # Сохраняем
        profile_path = ROOT / "data" / "profile.json"
        save_data = {k: v for k, v in profile.items() if not k.startswith("_")}
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        # Отвечаем
        text = (
            f"✅ *Резюме обработано!*\n\n"
            f"👤 Имя: {profile.get('name','—')}\n"
            f"🛠 Hard Skills: {', '.join(profile.get('hard_skills',[])[:8]) or '—'}\n"
            f"🤝 Soft Skills: {', '.join(profile.get('soft_skills',[])[:4]) or '—'}\n\n"
            f"Профиль сохранён. Используй /predict или /jobs для анализа."
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    except RuntimeError as e:
        await update.message.reply_text(
            f"⚠️ PDF-библиотека не установлена.\n`pip install pdfplumber`\n\n{e}",
            parse_mode="Markdown",
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка обработки PDF: {e}")


async def handle_text(update, context):
    """Обрабатывает произвольный текст как текстовое резюме."""
    text = update.message.text
    if len(text) < 50:
        await update.message.reply_text(
            "💬 Отправь текст резюме (или используй /start для списка команд)"
        )
        return

    await update.message.reply_text("📝 Обрабатываю текст как резюме...")
    try:
        from utils.resume_parser import parse_resume
        profile = parse_resume(text)

        profile_path = ROOT / "data" / "profile.json"
        save_data = {k: v for k, v in profile.items() if not k.startswith("_")}
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        resp = (
            f"✅ *Профиль обновлён!*\n\n"
            f"🛠 Hard Skills: {', '.join(profile.get('hard_skills',[])[:8]) or '—'}\n"
            f"🤝 Soft Skills: {', '.join(profile.get('soft_skills',[])[:4]) or '—'}\n\n"
            f"Используй /predict или /jobs"
        )
        await update.message.reply_text(resp, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


# ---------------------------------------------------------------------------
# Запуск бота
# ---------------------------------------------------------------------------

def run_bot():
    """Запускает Telegram бота."""
    try:
        from telegram.ext import (
            ApplicationBuilder, CommandHandler,
            MessageHandler, filters,
        )
    except ImportError:
        raise RuntimeError(
            "Установи python-telegram-bot:\n"
            "pip install python-telegram-bot"
        )

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN не задан в .env\n"
            "Получи токен у @BotFather в Telegram"
        )

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("profile", cmd_profile))
    app.add_handler(CommandHandler("predict", cmd_predict))
    app.add_handler(CommandHandler("jobs",    cmd_jobs))
    app.add_handler(CommandHandler("skills",  cmd_skills))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Бот запущен. Нажми Ctrl+C для остановки.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    run_bot()
