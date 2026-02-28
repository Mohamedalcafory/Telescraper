from pymongo import MongoClient
from pymongo.database import Database
from typing import Optional
import logging

from config import settings

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        logger.info(f"Connecting to MongoDB at {settings.MONGODB_URI}")
        _client = MongoClient(settings.MONGODB_URI)
        logger.info("MongoDB connection established")
    return _client


def get_database() -> Database:
    global _db
    if _db is None:
        client = get_client()
        _db = client[settings.MONGODB_DATABASE]
        logger.info(f"Using database: {settings.MONGODB_DATABASE}")
    return _db


def close_connection():
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed")


def ping() -> bool:
    try:
        client = get_client()
        client.admin.command('ping')
        return True
    except Exception as e:
        logger.error(f"MongoDB ping failed: {e}")
        return False
