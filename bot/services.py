import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from database import (
    MessageRepository,
    CheckpointRepository,
    Message,
    MessageStatus,
    ping as db_ping,
)
from publisher import TelegramPublisher
from filter import MessageFilter

logger = logging.getLogger(__name__)


class BotService:
    def __init__(self):
        self.message_repo = MessageRepository()
        self.checkpoint_repo = CheckpointRepository()
        self.publisher = None
        self.filter = MessageFilter()

    def _get_publisher(self) -> TelegramPublisher:
        if self.publisher is None:
            self.publisher = TelegramPublisher()
        return self.publisher

    def check_database(self) -> bool:
        try:
            return db_ping()
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        try:
            return self.message_repo.get_stats()
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def get_latest_messages(
        self, limit: int = 10, status: Optional[MessageStatus] = None
    ) -> List[Message]:
        try:
            if status:
                return self.message_repo.get_messages_by_status(status, limit)

            cursor = self.message_repo.collection.find().sort("date", -1).limit(limit)
            return [Message.from_dict(doc) for doc in cursor]
        except Exception as e:
            logger.error(f"Failed to get latest messages: {e}")
            return []

    def get_genocidal_messages(self, limit: int = 10) -> List[Message]:
        try:
            return self.filter.get_genocidal_messages(limit)
        except Exception as e:
            logger.error(f"Failed to get genocidal messages: {e}")
            return []

    def get_pending_messages(self, limit: int = 10) -> List[Message]:
        try:
            return self.message_repo.get_messages_by_status(
                MessageStatus.PENDING_PUBLISH, limit
            )
        except Exception as e:
            logger.error(f"Failed to get pending messages: {e}")
            return []

    def get_classified_messages(self, limit: int = 10) -> List[Message]:
        try:
            return self.message_repo.get_messages_by_status(
                MessageStatus.CLASSIFIED, limit
            )
        except Exception as e:
            logger.error(f"Failed to get classified messages: {e}")
            return []

    def get_failed_messages(self, limit: int = 10) -> List[Message]:
        try:
            return self.message_repo.get_failed_messages(limit)
        except Exception as e:
            logger.error(f"Failed to get failed messages: {e}")
            return []

    def get_message_by_id(self, message_id: int, channel: str) -> Optional[Message]:
        try:
            return self.message_repo.get_by_id(message_id, channel)
        except Exception as e:
            logger.error(f"Failed to get message {message_id}: {e}")
            return None

    def publish_message(self, message: Message) -> bool:
        try:
            pub = self._get_publisher()
            return pub.publish_message(message)
        except Exception as e:
            logger.error(f"Failed to publish message {message.message_id}: {e}")
            return False

    def publish_all_pending(self) -> int:
        try:
            messages = self.get_pending_messages(limit=50)
            if not messages:
                return 0

            pub = self._get_publisher()
            count = pub.publish_batch(messages)
            logger.info(f"Published {count} pending messages")
            return count
        except Exception as e:
            logger.error(f"Failed to publish all pending: {e}")
            return 0

    def retry_failed(self) -> int:
        try:
            pub = self._get_publisher()
            count = pub.retry_failed()
            logger.info(f"Retried {count} failed messages")
            return count
        except Exception as e:
            logger.error(f"Failed to retry: {e}")
            return 0

    def delete_message(self, message_id: int, channel: str) -> bool:
        try:
            return self.message_repo.delete_message(message_id, channel)
        except Exception as e:
            logger.error(f"Failed to delete message {message_id}: {e}")
            return False

    def get_checkpoint(self) -> Optional[Dict[str, Any]]:
        try:
            checkpoint = self.checkpoint_repo.get_checkpoint("muthanapress84")
            if checkpoint:
                return checkpoint.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get checkpoint: {e}")
            return None

    def get_all_checkpoints(self) -> List[Dict[str, Any]]:
        try:
            cursor = self.checkpoint_repo.collection.find()
            return [doc for doc in cursor]
        except Exception as e:
            logger.error(f"Failed to get all checkpoints: {e}")
            return []

    def get_config_summary(self) -> Dict[str, Any]:
        from config import settings
        from bot.config import bot_settings

        return {
            "source_channel": settings.SOURCE_CHANNEL,
            "publish_channel": settings.PUBLISH_CHANNEL,
            "mongodb_database": settings.MONGODB_DATABASE,
            "batch_size": settings.BATCH_SIZE,
            "max_retry": settings.MAX_RETRY_COUNT,
            "sync_interval_hours": settings.SYNC_INTERVAL_HOURS,
            "classification_model": settings.CLASSIFICATION_MODEL,
            "admin_users": len(bot_settings.admin_ids),
        }

    def search_messages(self, query: str, limit: int = 20) -> List[Message]:
        try:
            cursor = self.message_repo.collection.find(
                {
                    "$or": [
                        {"text": {"$regex": query, "$options": "i"}},
                        {
                            "classification.explanation": {
                                "$regex": query,
                                "$options": "i",
                            }
                        },
                    ]
                }
            ).limit(limit)
            return [Message.from_dict(doc) for doc in cursor]
        except Exception as e:
            logger.error(f"Failed to search messages: {e}")
            return []

    def get_messages_by_date_range(
        self, start_date: datetime, end_date: datetime, limit: int = 50
    ) -> List[Message]:
        try:
            cursor = (
                self.message_repo.collection.find(
                    {"date": {"$gte": start_date, "$lte": end_date}}
                )
                .sort("date", -1)
                .limit(limit)
            )
            return [Message.from_dict(doc) for doc in cursor]
        except Exception as e:
            logger.error(f"Failed to get messages by date range: {e}")
            return []

    def get_genocidal_count(self) -> int:
        try:
            return self.message_repo.collection.count_documents(
                {"classification.is_genocidal": True}
            )
        except Exception as e:
            logger.error(f"Failed to get genocidal count: {e}")
            return 0
