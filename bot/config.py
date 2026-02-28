import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "config" / ".env")


class BotSettings:
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    PUBLISH_CHANNEL: str = os.getenv("PUBLISH_CHANNEL", "")
    SOURCE_CHANNEL: str = os.getenv("SOURCE_CHANNEL", "muthanapress84")

    ADMIN_USER_IDS: str = os.getenv("ADMIN_USER_IDS", "")

    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "telescraper")

    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "50"))
    MAX_RETRY_COUNT: int = int(os.getenv("MAX_RETRY_COUNT", "3"))

    ENABLE_COMMANDS: bool = os.getenv("BOT_ENABLE_COMMANDS", "true").lower() == "true"
    ENABLE_CALLBACKS: bool = os.getenv("BOT_ENABLE_CALLBACKS", "true").lower() == "true"

    WELCOME_MESSAGE: str = os.getenv("BOT_WELCOME_MESSAGE", "")
    HELP_MESSAGE: str = os.getenv("BOT_HELP_MESSAGE", "")

    @property
    def admin_ids(self) -> list:
        if not self.ADMIN_USER_IDS:
            return []
        return [
            int(uid.strip()) for uid in self.ADMIN_USER_IDS.split(",") if uid.strip()
        ]

    @classmethod
    def validate(cls) -> bool:
        if not cls.TELEGRAM_BOT_TOKEN:
            print("Missing TELEGRAM_BOT_TOKEN in config/.env")
            return False
        return True

    @classmethod
    def is_admin(cls, user_id: int) -> bool:
        admin_ids = cls.admin_ids
        if not admin_ids:
            return True
        return user_id in admin_ids


bot_settings = BotSettings()
