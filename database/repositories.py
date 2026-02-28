from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from database.connection import get_database
from database.models import Message, Classification, MessageStatus, SyncCheckpoint

logger = logging.getLogger(__name__)


class MessageRepository:
    def __init__(self):
        self.collection = get_database()["messages"]

    def insert_message(self, message: Message) -> bool:
        try:
            self.collection.insert_one(message.to_dict())
            return True
        except Exception as e:
            logger.error(f"Failed to insert message: {e}")
            return False

    def insert_messages_bulk(self, messages: List[Message]) -> int:
        try:
            if not messages:
                return 0
            docs = [msg.to_dict() for msg in messages]
            result = self.collection.insert_many(docs)
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"Failed to bulk insert messages: {e}")
            return 0

    def message_exists(self, message_id: int, channel: str) -> bool:
        try:
            return self.collection.count_documents({
                "message_id": message_id,
                "channel": channel
            }, limit=1) > 0
        except Exception as e:
            logger.error(f"Failed to check message existence: {e}")
            return False

    def get_by_id(self, message_id: int, channel: str) -> Optional[Message]:
        try:
            doc = self.collection.find_one({
                "message_id": message_id,
                "channel": channel
            })
            if doc:
                return Message.from_dict(doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get message: {e}")
            return None

    def get_messages_by_status(self, status: MessageStatus, limit: int = 50) -> List[Message]:
        try:
            cursor = self.collection.find({"status": status.value}).limit(limit)
            return [Message.from_dict(doc) for doc in cursor]
        except Exception as e:
            logger.error(f"Failed to get messages by status: {e}")
            return []

    def get_genocidal_pending(self, limit: int = 50) -> List[Message]:
        try:
            cursor = self.collection.find({
                "classification.is_genocidal": True,
                "status": MessageStatus.CLASSIFIED.value
            }).sort("date", -1).limit(limit)
            return [Message.from_dict(doc) for doc in cursor]
        except Exception as e:
            logger.error(f"Failed to get genocidal messages: {e}")
            return []

    def update_classification(self, message_id: int, channel: str, classification: Classification) -> bool:
        try:
            result = self.collection.update_one(
                {"message_id": message_id, "channel": channel},
                {
                    "$set": {
                        "classification": classification.to_dict(),
                        "status": MessageStatus.CLASSIFIED.value,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update classification: {e}")
            return False

    def update_status(self, message_id: int, channel: str, status: MessageStatus, 
                      error_message: Optional[str] = None, retry_count: Optional[int] = None) -> bool:
        try:
            update_data = {
                "status": status.value,
                "updated_at": datetime.utcnow()
            }
            if error_message is not None:
                update_data["error_message"] = error_message
            if retry_count is not None:
                update_data["retry_count"] = retry_count
            if status == MessageStatus.PUBLISHED:
                update_data["published_at"] = datetime.utcnow()
            
            result = self.collection.update_one(
                {"message_id": message_id, "channel": channel},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            return False

    def set_pending_publish(self, message_ids: List[int], channel: str) -> int:
        try:
            result = self.collection.update_many(
                {
                    "message_id": {"$in": message_ids},
                    "channel": channel,
                    "status": MessageStatus.CLASSIFIED.value
                },
                {
                    "$set": {
                        "status": MessageStatus.PENDING_PUBLISH.value,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            return result.modified_count
        except Exception as e:
            logger.error(f"Failed to set pending publish: {e}")
            return 0

    def delete_message(self, message_id: int, channel: str) -> bool:
        try:
            result = self.collection.delete_one({
                "message_id": message_id,
                "channel": channel
            })
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete message: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        try:
            pipeline = [
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            result = list(self.collection.aggregate(pipeline))
            stats = {item["_id"]: item["count"] for item in result}
            
            total = self.collection.count_documents({})
            stats["total"] = total
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def get_failed_messages(self, max_retries: int) -> List[Message]:
        try:
            cursor = self.collection.find({
                "status": MessageStatus.FAILED.value,
                "retry_count": {"$lt": max_retries}
            }).limit(50)
            return [Message.from_dict(doc) for doc in cursor]
        except Exception as e:
            logger.error(f"Failed to get failed messages: {e}")
            return []


class CheckpointRepository:
    def __init__(self):
        self.collection = get_database()["checkpoints"]

    def get_checkpoint(self, channel: str) -> Optional[SyncCheckpoint]:
        try:
            doc = self.collection.find_one({"channel": channel})
            if doc:
                return SyncCheckpoint.from_dict(doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get checkpoint: {e}")
            return None

    def save_checkpoint(self, checkpoint: SyncCheckpoint) -> bool:
        try:
            self.collection.update_one(
                {"channel": checkpoint.channel},
                {"$set": checkpoint.to_dict()},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False


class SettingsRepository:
    def __init__(self):
        self.collection = get_database()["settings"]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            doc = self.collection.find_one({"key": key})
            if doc:
                return doc.get("value", default)
            return default
        except Exception as e:
            logger.error(f"Failed to get setting: {e}")
            return default

    def set(self, key: str, value: Any) -> bool:
        try:
            self.collection.update_one(
                {"key": key},
                {"$set": {"value": value, "updated_at": datetime.utcnow()}},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set setting: {e}")
            return False
