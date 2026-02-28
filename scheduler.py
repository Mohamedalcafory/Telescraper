import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime

from config import settings
from orchestrator import PipelineOrchestrator

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline():
    logger.info("=" * 60)
    logger.info(f"SCHEDULED PIPELINE RUN - {datetime.utcnow()}")
    logger.info("=" * 60)
    
    orchestrator = PipelineOrchestrator()
    import asyncio
    asyncio.run(orchestrator.run_full_pipeline())
    
    logger.info("=" * 60)
    logger.info("SCHEDULED RUN COMPLETE")
    logger.info("=" * 60)


def main():
    logger.info(f"Starting scheduler with {settings.SYNC_INTERVAL_HOURS} hour interval")
    
    scheduler = BlockingScheduler()
    
    scheduler.add_job(
        run_pipeline,
        trigger=IntervalTrigger(hours=settings.SYNC_INTERVAL_HOURS),
        id='pipeline_job',
        name='Telegram Pipeline',
        replace_existing=True
    )
    
    logger.info("Scheduler started. Press Ctrl+C to stop.")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
