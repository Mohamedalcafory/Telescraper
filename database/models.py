from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class MessageStatus(str, Enum):
    NEW = "new"
    CLASSIFIED = "classified"
    PENDING_PUBLISH = "pending_publish"
    PUBLISHED = "published"
    FAILED = "failed"


@dataclass
class Classification:
    civilian_deaths: bool = False
    targeting_civilians: bool = False
    blocking_aid: bool = False
    destroying_homes: bool = False
    targeting_facilities: bool = False
    forced_displacement: bool = False
    systematic_violence: bool = False
    is_official_speech: bool = False
    is_genocidal: bool = False
    explanation: str = ""
    classified_at: Optional[datetime] = None
    model_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "civilian_deaths": self.civilian_deaths,
            "targeting_civilians": self.targeting_civilians,
            "blocking_aid": self.blocking_aid,
            "destroying_homes": self.destroying_homes,
            "targeting_facilities": self.targeting_facilities,
            "forced_displacement": self.forced_displacement,
            "systematic_violence": self.systematic_violence,
            "is_official_speech": self.is_official_speech,
            "is_genocidal": self.is_genocidal,
            "explanation": self.explanation,
            "classified_at": self.classified_at,
            "model_used": self.model_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Classification":
        return cls(
            civilian_deaths=data.get("civilian_deaths", False),
            targeting_civilians=data.get("targeting_civilians", False),
            blocking_aid=data.get("blocking_aid", False),
            destroying_homes=data.get("destroying_homes", False),
            targeting_facilities=data.get("targeting_facilities", False),
            forced_displacement=data.get("forced_displacement", False),
            systematic_violence=data.get("systematic_violence", False),
            is_official_speech=data.get("is_official_speech", False),
            is_genocidal=data.get("is_genocidal", False),
            explanation=data.get("explanation", ""),
            classified_at=data.get("classified_at"),
            model_used=data.get("model_used", ""),
        )


@dataclass
class Message:
    message_id: int
    channel: str
    text: str
    date: datetime
    url: str
    classification: Optional[Classification] = None
    status: MessageStatus = MessageStatus.NEW
    retry_count: int = 0
    error_message: Optional[str] = None
    published_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "message_id": self.message_id,
            "channel": self.channel,
            "text": self.text,
            "date": self.date,
            "url": self.url,
            "status": self.status.value if isinstance(self.status, MessageStatus) else self.status,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "published_at": self.published_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.classification:
            data["classification"] = self.classification.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        classification = None
        if data.get("classification"):
            classification = Classification.from_dict(data["classification"])
        
        status = data.get("status", "new")
        if isinstance(status, str):
            status = MessageStatus(status)
        
        return cls(
            message_id=data["message_id"],
            channel=data["channel"],
            text=data["text"],
            date=data["date"],
            url=data["url"],
            classification=classification,
            status=status,
            retry_count=data.get("retry_count", 0),
            error_message=data.get("error_message"),
            published_at=data.get("published_at"),
            created_at=data.get("created_at", datetime.utcnow()),
            updated_at=data.get("updated_at", datetime.utcnow()),
        )


@dataclass
class SyncCheckpoint:
    channel: str
    last_message_id: int
    last_sync: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "last_message_id": self.last_message_id,
            "last_sync": self.last_sync,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncCheckpoint":
        return cls(
            channel=data["channel"],
            last_message_id=data["last_message_id"],
            last_sync=data.get("last_sync", datetime.utcnow()),
        )
