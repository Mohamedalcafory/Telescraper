#!/usr/bin/env python3
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings


def main():
    parser = argparse.ArgumentParser(
        description='TeleScrape - Telegram Pipeline for Genocide Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --run-once           Run pipeline once
  python main.py --schedule           Run pipeline on schedule
  python main.py --sync               Sync only
  python main.py --classify           Classify only  
  python main.py --publish             Publish only
  python main.py --stats              Show database statistics
        """
    )
    
    parser.add_argument('--run-once', action='store_true',
                        help='Run the full pipeline once')
    parser.add_argument('--schedule', action='store_true',
                        help='Run the pipeline on a schedule')
    parser.add_argument('--sync', action='store_true',
                        help='Sync messages from Telegram only')
    parser.add_argument('--classify', action='store_true',
                        help='Classify messages only')
    parser.add_argument('--publish', action='store_true',
                        help='Publish messages only')
    parser.add_argument('--stats', action='store_true',
                        help='Show database statistics')
    
    args = parser.parse_args()
    
    if not settings.validate():
        print("\nPlease configure config/.env file before running.")
        print("Copy config/.env.example to config/.env and fill in your values.")
        sys.exit(1)
    
    if args.schedule:
        from scheduler import main as scheduler_main
        scheduler_main()
    elif args.run_once:
        from orchestrator import PipelineOrchestrator
        import asyncio
        orchestrator = PipelineOrchestrator()
        asyncio.run(orchestrator.run_full_pipeline())
    elif args.sync:
        from orchestrator import PipelineOrchestrator
        orch = PipelineOrchestrator()
        orch.run_sync_only()
    elif args.classify:
        from orchestrator import PipelineOrchestrator
        orch = PipelineOrchestrator()
        orch.run_classify_only()
    elif args.publish:
        from orchestrator import PipelineOrchestrator
        orch = PipelineOrchestrator()
        orch.run_publish_only()
    elif args.stats:
        from database import MessageRepository
        repo = MessageRepository()
        stats = repo.get_stats()
        print("\n=== Database Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
