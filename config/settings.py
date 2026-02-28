import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / "config" / ".env")


class Settings:
    # MongoDB
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "telescraper")

    # Telegram Scraper
    TELEGRAM_API_ID: int = int(os.getenv("TELEGRAM_API_ID", "26874679"))
    TELEGRAM_API_HASH: str = os.getenv("TELEGRAM_API_HASH", "")
    TELEGRAM_PHONE: str = os.getenv("TELEGRAM_PHONE", "")
    SOURCE_CHANNEL: str = os.getenv("SOURCE_CHANNEL", "muthanapress84")

    # Telegram Publisher
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    PUBLISH_CHANNEL: str = os.getenv("PUBLISH_CHANNEL", "")

    # OpenRouter
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

    # Pipeline Settings
    SYNC_INTERVAL_HOURS: int = int(os.getenv("SYNC_INTERVAL_HOURS", "2"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "50"))
    MAX_RETRY_COUNT: int = int(os.getenv("MAX_RETRY_COUNT", "3"))
    RETRY_DELAY_SECONDS: int = int(os.getenv("RETRY_DELAY_SECONDS", "60"))
    CLASSIFICATION_MODEL: str = os.getenv("CLASSIFICATION_MODEL", "deepseek/deepseek-chat-v3-0324:free")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls) -> bool:
        required = [
            ("MONGODB_URI", cls.MONGODB_URI),
            ("TELEGRAM_API_ID", cls.TELEGRAM_API_ID),
            ("TELEGRAM_API_HASH", cls.TELEGRAM_API_HASH),
            ("TELEGRAM_PHONE", cls.TELEGRAM_PHONE),
            ("SOURCE_CHANNEL", cls.SOURCE_CHANNEL),
            ("TELEGRAM_BOT_TOKEN", cls.TELEGRAM_BOT_TOKEN),
            ("PUBLISH_CHANNEL", cls.PUBLISH_CHANNEL),
            ("OPENROUTER_API_KEY", cls.OPENROUTER_API_KEY),
        ]
        
        missing = []
        for name, value in required:
            if not value:
                missing.append(name)
        
        if missing:
            print(f"Missing required configuration: {', '.join(missing)}")
            print("Please update config/.env file")
            return False
        return True


settings = Settings()
