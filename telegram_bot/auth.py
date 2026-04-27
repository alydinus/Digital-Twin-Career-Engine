"""
Telegram — одноразовая авторизация.

Запускается ОДИН РАЗ из терминала для создания .session файла:

    python telegram_bot/auth.py

Что делает:
  1. Берёт ключи из .env (TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_PHONE)
  2. Отправляет SMS-код на указанный номер
  3. Ты вводишь код → сессия сохраняется в data/.telegram_session.session
  4. После этого Streamlit-приложение использует эту сессию без повторного ввода кода

Требования:
  pip install telethon python-dotenv
"""

import sys
import asyncio
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Загружаем .env ────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    print("✅ .env загружен")
except ImportError:
    print("⚠️  python-dotenv не установлен: pip install python-dotenv")

import os

API_ID    = os.environ.get("TELEGRAM_API_ID", "").strip()
API_HASH  = os.environ.get("TELEGRAM_API_HASH", "").strip()
PHONE     = os.environ.get("TELEGRAM_PHONE", "").strip()

# ── Проверка конфига ──────────────────────────────────────────────────────────
print("\n── Проверка переменных окружения ──")
print(f"  TELEGRAM_API_ID   = {'✅ ' + API_ID[:4] + '...' if API_ID else '❌ не задан'}")
print(f"  TELEGRAM_API_HASH = {'✅ ' + API_HASH[:4] + '...' if API_HASH else '❌ не задан'}")
print(f"  TELEGRAM_PHONE    = {'✅ ' + PHONE if PHONE else '❌ не задан'}")

if not (API_ID and API_HASH and PHONE):
    print("\n❌ Заполни TELEGRAM_API_ID, TELEGRAM_API_HASH и TELEGRAM_PHONE в .env")
    print("   Ключи получить на: https://my.telegram.org/apps")
    sys.exit(1)

# ── Авторизация ───────────────────────────────────────────────────────────────
SESSION_PATH = ROOT / "data" / ".telegram_session"


async def authorize():
    try:
        from telethon import TelegramClient
        from telethon.errors import SessionPasswordNeededError
    except ImportError:
        print("\n❌ Telethon не установлен. Запусти: pip install telethon")
        sys.exit(1)

    print(f"\n🔗 Подключаюсь к Telegram как {PHONE}...")
    client = TelegramClient(str(SESSION_PATH), int(API_ID), API_HASH)

    await client.connect()

    if await client.is_user_authorized():
        me = await client.get_me()
        print(f"\n✅ Уже авторизован как @{me.username} ({me.first_name})")
        print(f"   Сессия: {SESSION_PATH}.session")
        await client.disconnect()
        return

    # Отправляем OTP
    await client.send_code_request(PHONE)
    print(f"\n📱 SMS-код отправлен на {PHONE}")
    code = input("Введи код из SMS: ").strip()

    try:
        await client.sign_in(PHONE, code)
    except SessionPasswordNeededError:
        # Двухфакторная авторизация (2FA)
        password = input("Введи пароль двухфакторной аутентификации: ").strip()
        await client.sign_in(password=password)

    me = await client.get_me()
    print(f"\n✅ Авторизован как @{me.username} ({me.first_name})")
    print(f"   Сессия сохранена: {SESSION_PATH}.session")
    print("\n🚀 Теперь можешь запускать Streamlit — реальный поиск вакансий активен.")
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(authorize())
