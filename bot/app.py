import os
import sys
import logging
from pathlib import Path

import telegram
from telegram.ext import Application

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from bot.config import bot_settings
from bot.handlers import setup_handlers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TeleScrapeBot:
    def __init__(self):
        self.token = bot_settings.TELEGRAM_BOT_TOKEN
        self.application = None

    def validate_config(self) -> bool:
        if not self.token:
            logger.error("TELEGRAM_BOT_TOKEN not configured")
            return False

        if not bot_settings.validate():
            logger.error("Bot configuration validation failed")
            return False

        return True

    async def post_init(self, application: Application):
        bot = application.bot

        try:
            bot_info = await bot.get_me()
            logger.info(f"Bot started: @{bot_info.username} ({bot_info.name})")

            await bot.set_my_commands(
                [
                    ("start", "بدء استخدام البوت"),
                    ("latest", "أحدث الرسائل"),
                    ("stats", "الإحصائيات"),
                    ("pending", "الرسائل المعلقة"),
                    ("failed", "الرسائل الفاشلة"),
                    ("genocidal", "رسائل الإبادة الجماعية"),
                    ("help", "المساعدة"),
                    ("admin", "لوحة الإدارة"),
                ]
            )
            logger.info("Bot commands updated")

        except Exception as e:
            logger.error(f"Post-init error: {e}")

    def run(self):
        if not self.validate_config():
            logger.error("Bot configuration validation failed. Exiting.")
            sys.exit(1)

        logger.info("Starting TeleScrape Bot...")

        self.application = (
            Application.builder().token(self.token).post_init(self.post_init).build()
        )

        setup_handlers(self.application)

        logger.info("Bot handlers configured. Starting polling...")

        self.application.run_polling(
            drop_pending_updates=True, allowed_updates=["message", "callback_query"]
        )

    async def stop(self):
        if self.application:
            await self.application.stop()
            logger.info("Bot stopped")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TeleScrape Bot")
    parser.add_argument("--poll", action="store_true", help="Run bot in polling mode")
    parser.add_argument(
        "--webhook", action="store_true", help="Run bot in webhook mode"
    )
    parser.add_argument("--port", type=int, default=8443, help="Webhook port")
    parser.add_argument("--url", type=str, default="", help="Webhook URL")
    args = parser.parse_args()

    bot = TeleScrapeBot()

    if args.webhook:
        if not args.url:
            logger.error("Webhook URL required for webhook mode")
            sys.exit(1)

        bot.application = (
            Application.builder().token(bot.token).post_init(bot.post_init).build()
        )

        setup_handlers(bot.application)

        logger.info(f"Starting webhook on {args.url}:{args.port}")
        bot.application.run_webhook(
            listen="0.0.0.0",
            port=args.port,
            url_path="webhook",
            webhook_url=args.url,
            allowed_updates=["message", "callback_query"],
        )
    else:
        bot.run()


if __name__ == "__main__":
    main()
