from typing import List
import logging

from database import Message, MessageRepository, MessageStatus

logger = logging.getLogger(__name__)


class MessageFilter:
    def __init__(self):
        self.message_repo = MessageRepository()
    
    def get_genocidal_messages(self, limit: int = 50) -> List[Message]:
        messages = self.message_repo.get_genocidal_pending(limit)
        
        if messages:
            message_ids = [msg.message_id for msg in messages]
            channel = messages[0].channel
            self.message_repo.set_pending_publish(message_ids, channel)
            logger.info(f"Marked {len(messages)} messages as pending publish")
        
        return messages
    
    def get_pending_retry(self, max_retries: int) -> List[Message]:
        return self.message_repo.get_failed_messages(max_retries)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    filter = MessageFilter()
    messages = filter.get_genocidal_messages()
    print(f"Found {len(messages)} genocidal messages to publish")


if __name__ == "__main__":
    main()
