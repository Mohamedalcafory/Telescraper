import logging
import time
from typing import List, Optional

import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from config import settings
from database import Message, MessageRepository, MessageStatus

logger = logging.getLogger(__name__)


class MessageFormatter:
    @staticmethod
    def format_message(message: Message) -> str:
        classification = message.classification
        
        flags = []
        if classification:
            if classification.civilian_deaths:
                flags.append("👥 قتل مدنيين")
            if classification.targeting_civilians:
                flags.append("🎯 استهداف مدنيين")
            if classification.destroying_homes:
                flags.append("🏠 تدمير منازل")
            if classification.targeting_facilities:
                flags.append("🏥 استهداف منشآت")
            if classification.forced_displacement:
                flags.append("🚶 تهجير قسري")
            if classification.blocking_aid:
                flags.append("🚫 عرقلة مساعدات")
        
        flags_str = "\n".join(flags) if flags else "بدون تصنيف محدد"
        
        formatted = f"""🔴 تقرير إبادة جماعية

{message.text}

────────────────────
{flags_str}

📅 {message.date.strftime('%Y-%m-%d %H:%M')}
📢 المصدر: @{message.channel}
🔗 [رابط الرسالة]({message.url})"""
        
        return formatted
    
    @staticmethod
    def format_short(message: Message) -> str:
        text = message.text[:500]
        if len(message.text) > 500:
            text += "..."
        
        return f"""🔴 #{message.message_id}

{text}

📅 {message.date.strftime('%Y-%m-%d')}
🔗 {message.url}"""


class TelegramPublisher:
    def __init__(self):
        self.bot = telegram.Bot(token=settings.TELEGRAM_BOT_TOKEN)
        self.chat_id = settings.PUBLISH_CHANNEL
        self.message_repo = MessageRepository()
        self.max_retries = settings.MAX_RETRY_COUNT
        self.retry_delay = settings.RETRY_DELAY_SECONDS
    
    async def send_message(self, message: Message, formatted_text: str = None) -> bool:
        if formatted_text is None:
            formatted_text = MessageFormatter.format_message(message)
        
        try:
            msg = await self.bot.send_message(
                chat_id=self.chat_id,
                text=formatted_text,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
            logger.info(f"Sent message {message.message_id}, Telegram msg_id: {msg.message_id}")
            return True
        except telegram.error.TelegramError as e:
            logger.error(f"Failed to send message {message.message_id}: {e}")
            return False
    
    def publish_message(self, message: Message) -> bool:
        formatted = MessageFormatter.format_message(message)
        
        for attempt in range(self.max_retries):
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(self.send_message(message, formatted))
                loop.close()
                
                if success:
                    self.message_repo.update_status(
                        message.message_id,
                        message.channel,
                        MessageStatus.PUBLISHED
                    )
                    
                    self.message_repo.delete_message(message.message_id, message.channel)
                    logger.info(f"Published and deleted message {message.message_id}")
                    return True
                
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        retry_count = message.retry_count + 1
        self.message_repo.update_status(
            message.message_id,
            message.channel,
            MessageStatus.FAILED,
            error_message=f"Failed after {self.max_retries} attempts",
            retry_count=retry_count
        )
        
        logger.error(f"Message {message.message_id} marked as failed after {self.max_retries} attempts")
        return False
    
    def publish_batch(self, messages: List[Message]) -> int:
        published = 0
        for message in messages:
            if self.publish_message(message):
                published += 1
            time.sleep(1)
        
        logger.info(f"Published {published}/{len(messages)} messages")
        return published
    
    def retry_failed(self) -> int:
        failed_messages = self.message_repo.get_failed_messages(self.max_retries)
        
        if not failed_messages:
            logger.info("No failed messages to retry")
            return 0
        
        logger.info(f"Retrying {len(failed_messages)} failed messages")
        
        for msg in failed_messages:
            msg.retry_count = 0
            self.publish_message(msg)
        
        return len(failed_messages)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    from filter import MessageFilter
    
    filter = MessageFilter()
    messages = filter.get_genocidal_messages()
    
    if not messages:
        print("No messages to publish")
        return
    
    publisher = TelegramPublisher()
    count = publisher.publish_batch(messages)
    print(f"Published {count} messages")


if __name__ == "__main__":
    import sys
    try:
        import asyncio
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
