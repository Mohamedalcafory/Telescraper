from telethon import TelegramClient, errors
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import random
import time

from config import settings
from database import Message, MessageRepository, CheckpointRepository, SyncCheckpoint

logger = logging.getLogger(__name__)

SYNC_HOURS = 2  # Sync messages from last 2 hours


class TelegramScraper:
    def __init__(self):
        self.client = TelegramClient(
            "my_telegram_session", settings.TELEGRAM_API_ID, settings.TELEGRAM_API_HASH
        )
        self.channel_username = settings.SOURCE_CHANNEL
        self.message_repo = MessageRepository()
        self.checkpoint_repo = CheckpointRepository()

    async def authorize(self, phone: str):
        try:
            await self.client.start(phone)
            if not await self.client.is_user_authorized():
                await self.client.send_code_request(phone)
                code = input("Enter the code: ")
                await self.client.sign_in(phone, code)
            me = await self.client.get_me()
            logger.info(f"Connected as: {me.first_name} (@{me.username})")
            return True
        except errors.FloodWaitError as e:
            logger.error(f"Rate limited. Need to wait {e.seconds} seconds.")
            await asyncio.sleep(e.seconds)
            return await self.authorize(phone)
        except Exception as e:
            logger.error(f"Authorization failed: {e}")
            return False

    async def retry_with_backoff(self, coro, max_retries=5, initial_delay=1):
        retries = 0
        delay = initial_delay

        while True:
            try:
                return await coro
            except errors.FloodWaitError as e:
                wait_time = e.seconds
                logger.warning(f"Rate limited. Waiting {wait_time} seconds.")
                await asyncio.sleep(wait_time)
                retries = 0
                delay = initial_delay
            except (errors.ServerError, errors.TimedOutError, ConnectionError) as e:
                if retries >= max_retries:
                    logger.error(f"Max retries reached: {e}")
                    raise
                jitter = random.uniform(0.1, 0.5)
                sleep_time = delay + jitter
                logger.warning(
                    f"Attempt {retries + 1} failed: {e}. Retrying in {sleep_time:.2f}s"
                )
                await asyncio.sleep(sleep_time)
                retries += 1
                delay *= 2

    async def sync_messages(self, limit: int = 100) -> int:
        channel = await self.retry_with_backoff(
            self.client.get_entity(self.channel_username)
        )

        checkpoint = self.checkpoint_repo.get_checkpoint(self.channel_username)
        offset_id = checkpoint.last_message_id if checkpoint else 0

        since_date = datetime.now(timezone.utc) - timedelta(hours=SYNC_HOURS)

        logger.info(
            f"Syncing messages from {self.channel_username} (last {SYNC_HOURS} hours), offset_id: {offset_id}"
        )

        new_messages: List[Message] = []
        fetched = 0
        newest_id = offset_id

        while fetched < limit:
            current_batch = min(100, limit - fetched)

            messages = await self.retry_with_backoff(
                self.client.get_messages(
                    channel, limit=current_batch, offset_id=offset_id, reverse=True
                )
            )

            if not messages:
                break

            for msg in messages:
                if msg.id <= offset_id:
                    continue

                if msg.date < since_date:
                    logger.info(
                        f"Reached messages older than {SYNC_HOURS} hours, stopping"
                    )
                    break

                if not self.message_repo.message_exists(msg.id, self.channel_username):
                    message = Message(
                        message_id=msg.id,
                        channel=self.channel_username,
                        text=msg.text or "",
                        date=msg.date,
                        url=f"https://t.me/{self.channel_username}/{msg.id}",
                    )
                    new_messages.append(message)

                if msg.id > newest_id:
                    newest_id = msg.id

            fetched += len(messages)

            if messages[-1].date < since_date:
                break

            await asyncio.sleep(random.uniform(1, 3))

        if new_messages:
            inserted = self.message_repo.insert_messages_bulk(new_messages)
            logger.info(f"Inserted {inserted} new messages")
        else:
            logger.info("No new messages found")

        new_checkpoint = SyncCheckpoint(
            channel=self.channel_username, last_message_id=newest_id
        )
        self.checkpoint_repo.save_checkpoint(new_checkpoint)

        return len(new_messages)

    async def run(self):
        authorized = await self.authorize(settings.TELEGRAM_PHONE)
        if not authorized:
            logger.error("Failed to authorize")
            return 0

        return await self.sync_messages(settings.BATCH_SIZE)


async def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    scraper = TelegramScraper()
    async with scraper.client:
        count = await scraper.run()
        print(f"Synced {count} messages")


if __name__ == "__main__":
    import sys
    from telethon.errors import ConnectionError

    try:
        asyncio.run(main())
    except ConnectionError as e:
        print(f"Connection error: {e}")
        sys.exit(1)
