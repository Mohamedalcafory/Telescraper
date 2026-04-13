import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from database import Message, MessageRepository, MessageStatus
from config import settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 50
PUBLISH_INTERVAL_HOURS = 2


@dataclass
class MessageBatch:
    messages: List[Message]
    created_at: datetime = field(default_factory=datetime.utcnow)
    batch_id: str = field(
        default_factory=lambda: datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    )

    @property
    def count(self) -> int:
        return len(self.messages)

    @property
    def is_full(self) -> bool:
        return self.count >= BATCH_SIZE

    @property
    def is_stale(self) -> bool:
        age = datetime.utcnow() - self.created_at
        return age.total_seconds() >= (PUBLISH_INTERVAL_HOURS * 3600)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "message_ids": [msg.message_id for msg in self.messages],
            "count": self.count,
            "created_at": self.created_at,
        }


class MessageAggregator:
    def __init__(self):
        self.message_repo = MessageRepository()
        self.current_batch: Optional[MessageBatch] = None
        self.last_publish_time: Optional[datetime] = None

    def should_create_batch(self) -> bool:
        if self.current_batch is None:
            return True
        if self.current_batch.is_full:
            return True
        if self.current_batch.is_stale:
            return True
        return False

    def should_publish(self) -> bool:
        if self.current_batch is None:
            return False
        if self.current_batch.is_full:
            return True
        if self.current_batch.is_stale:
            return True
        if self.last_publish_time is None:
            return self.current_batch.count > 0

        time_since_last = datetime.utcnow() - self.last_publish_time
        return time_since_last.total_seconds() >= (PUBLISH_INTERVAL_HOURS * 3600)

    def collect_genocidal_messages(self, limit: int = BATCH_SIZE) -> List[Message]:
        try:
            from filter import MessageFilter

            filter_obj = MessageFilter()
            messages = filter_obj.get_genocidal_messages(limit)

            for msg in messages:
                self.message_repo.update_status(
                    msg.message_id, msg.channel, MessageStatus.PENDING_PUBLISH
                )

            logger.info(f"Collected {len(messages)} genocidal messages for aggregation")
            return messages
        except Exception as e:
            logger.error(f"Failed to collect genocidal messages: {e}")
            return []

    def create_batch(self, messages: List[Message]) -> MessageBatch:
        batch = MessageBatch(messages=messages)
        self.current_batch = batch
        logger.info(
            f"Created new batch with {batch.count} messages (batch_id: {batch.batch_id})"
        )
        return batch

    def get_batch_for_publishing(self) -> Optional[MessageBatch]:
        if not self.should_publish():
            return None

        batch = self.current_batch
        self.current_batch = None
        self.last_publish_time = datetime.utcnow()

        if batch:
            logger.info(
                f"Publishing batch {batch.batch_id} with {batch.count} messages"
            )

        return batch

    def get_pending_count(self) -> int:
        if self.current_batch:
            return self.current_batch.count
        return 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "current_batch_count": self.get_pending_count(),
            "batch_size": BATCH_SIZE,
            "publish_interval_hours": PUBLISH_INTERVAL_HOURS,
            "last_publish_time": self.last_publish_time,
            "is_batch_full": self.current_batch.is_full
            if self.current_batch
            else False,
            "is_batch_stale": self.current_batch.is_stale
            if self.current_batch
            else False,
        }


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    aggregator = MessageAggregator()
    messages = aggregator.collect_genocidal_messages()
    print(f"Collected {len(messages)} messages")

    if messages:
        batch = aggregator.create_batch(messages)
        print(f"Created batch: {batch.to_dict()}")


if __name__ == "__main__":
    main()
