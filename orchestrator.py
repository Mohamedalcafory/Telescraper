import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from config import settings
from database import ping, MessageRepository
from scraper import TelegramScraper
from classifier import MessageClassifier
from filter import MessageFilter
from publisher import TelegramPublisher

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(self):
        self.stats = {
            "scraped": 0,
            "classified": 0,
            "published": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None
        }
    
    def log_stats(self):
        logger.info("=" * 50)
        logger.info("PIPELINE STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Messages scraped: {self.stats['scraped']}")
        logger.info(f"Messages classified: {self.stats['classified']}")
        logger.info(f"Messages published: {self.stats['published']}")
        logger.info(f"Messages failed: {self.stats['failed']}")
        if self.stats['start_time']:
            logger.info(f"Started: {self.stats['start_time']}")
        if self.stats['end_time']:
            logger.info(f"Finished: {self.stats['end_time']}")
        logger.info("=" * 50)
    
    async def step_sync(self) -> int:
        logger.info("Step 1: Syncing messages from Telegram...")
        try:
            scraper = TelegramScraper()
            async with scraper.client:
                count = await scraper.sync_messages()
            self.stats['scraped'] = count
            logger.info(f"Synced {count} messages")
            return count
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return 0
    
    def step_classify(self) -> int:
        logger.info("Step 2: Classifying new messages...")
        try:
            classifier = MessageClassifier()
            count = classifier.process_batch()
            self.stats['classified'] = count
            logger.info(f"Classified {count} messages")
            return count
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return 0
    
    def step_filter(self):
        logger.info("Step 3: Filtering genocidal messages...")
        try:
            filter = MessageFilter()
            messages = filter.get_genocidal_messages()
            logger.info(f"Found {len(messages)} genocidal messages")
            return messages
        except Exception as e:
            logger.error(f"Filter failed: {e}")
            return []
    
    def step_publish(self, messages) -> int:
        logger.info("Step 4: Publishing messages...")
        try:
            publisher = TelegramPublisher()
            count = publisher.publish_batch(messages)
            self.stats['published'] = count
            logger.info(f"Published {count} messages")
            return count
        except Exception as e:
            logger.error(f"Publish failed: {e}")
            return 0
    
    def step_retry_failed(self) -> int:
        logger.info("Step 5: Retrying failed messages...")
        try:
            publisher = TelegramPublisher()
            count = publisher.retry_failed()
            logger.info(f"Retried {count} failed messages")
            return count
        except Exception as e:
            logger.error(f"Retry failed: {e}")
            return 0
    
    async def run_full_pipeline(self):
        self.stats['start_time'] = datetime.utcnow()
        logger.info("Starting full pipeline execution...")
        
        if not ping():
            logger.error("MongoDB connection failed")
            return
        
        await self.step_sync()
        
        self.step_classify()
        
        messages = self.step_filter()
        
        self.step_publish(messages)
        
        self.step_retry_failed()
        
        repo = MessageRepository()
        db_stats = repo.get_stats()
        logger.info(f"Database stats: {db_stats}")
        
        self.stats['end_time'] = datetime.utcnow()
        self.log_stats()
        
        return self.stats
    
    def run_sync_only(self):
        logger.info("Running sync-only mode...")
        self.stats['start_time'] = datetime.utcnow()
        
        if not ping():
            logger.error("MongoDB connection failed")
            return
        
        try:
            scraper = TelegramScraper()
            count = asyncio.run(scraper.run())
            self.stats['scraped'] = count
            logger.info(f"Synced {count} messages")
        except Exception as e:
            logger.error(f"Sync failed: {e}")
        
        self.stats['end_time'] = datetime.utcnow()
        self.log_stats()
    
    def run_classify_only(self):
        logger.info("Running classify-only mode...")
        self.stats['start_time'] = datetime.utcnow()
        
        if not ping():
            logger.error("MongoDB connection failed")
            return
        
        count = self.step_classify()
        
        self.stats['end_time'] = datetime.utcnow()
        self.log_stats()
        return count
    
    def run_publish_only(self):
        logger.info("Running publish-only mode...")
        self.stats['start_time'] = datetime.utcnow()
        
        if not ping():
            logger.error("MongoDB connection failed")
            return
        
        messages = self.step_filter()
        count = self.step_publish(messages)
        
        self.stats['end_time'] = datetime.utcnow()
        self.log_stats()
        return count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline Orchestrator')
    parser.add_argument('--mode', choices=['full', 'sync', 'classify', 'publish'], 
                        default='full', help='Pipeline mode')
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator()
    
    if args.mode == 'full':
        asyncio.run(orchestrator.run_full_pipeline())
    elif args.mode == 'sync':
        orchestrator.run_sync_only()
    elif args.mode == 'classify':
        orchestrator.run_classify_only()
    elif args.mode == 'publish':
        orchestrator.run_publish_only()


if __name__ == "__main__":
    main()
