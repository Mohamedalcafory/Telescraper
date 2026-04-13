import logging
from typing import List, Dict
from datetime import datetime
from collections import defaultdict

from database import Message

logger = logging.getLogger(__name__)


class DigestFormatter:
    MAX_DIGEST_LENGTH = 4000  # Telegram message limit is ~4096
    MAX_ITEM_LENGTH = 300

    @staticmethod
    def format_digest(messages: List[Message], batch_id: str = None) -> str:
        if not messages:
            return "📭 لا توجد رسائل للنشر"

        count = len(messages)
        date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

        lines = [
            f"🔴 تقرير إبادة جماعية - ملخص",
            f"",
            f"📅 {date_str}",
            f"📊 عدد الحوادث: {count}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"",
        ]

        # Group messages by classification flags
        categories = DigestFormatter._categorize_messages(messages)

        # Add category summaries
        if categories:
            lines.append("🏷️ *التصنيفات:*")
            for category, cat_messages in categories.items():
                lines.append(f"  • {category}: {len(cat_messages)}")
            lines.append("")
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append("")

        # Add message summaries
        for idx, msg in enumerate(messages, 1):
            item_text = DigestFormatter._format_message_item(msg, idx)
            lines.append(item_text)
            lines.append("")

        # Add footer
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        if batch_id:
            lines.append(f"🆔 معرف الدفعة: {batch_id}")
        lines.append(f"📢 المصدر: @{messages[0].channel if messages else 'unknown'}")

        digest = "\n".join(lines)

        # Truncate if too long
        if len(digest) > DigestFormatter.MAX_DIGEST_LENGTH:
            digest = digest[: DigestFormatter.MAX_DIGEST_LENGTH - 100]
            digest += "\n\n... (تم اقتصاص بعض المحتوى)"

        return digest

    @staticmethod
    def _format_message_item(msg: Message, index: int) -> str:
        text = msg.text[: DigestFormatter.MAX_ITEM_LENGTH]
        if len(msg.text) > DigestFormatter.MAX_ITEM_LENGTH:
            text += "..."

        flags = DigestFormatter._get_message_flags(msg)
        flags_str = " | ".join(flags) if flags else ""

        lines = [
            f"{index}. {text}",
        ]

        if flags_str:
            lines.append(f"   🏷️ {flags_str}")

        lines.append(f"   🔗 [رابط]({msg.url})")

        return "\n".join(lines)

    @staticmethod
    def _get_message_flags(msg: Message) -> List[str]:
        flags = []

        if msg.classification:
            if msg.classification.civilian_deaths:
                flags.append("👥 قتل مدنيين")
            if msg.classification.targeting_civilians:
                flags.append("🎯 استهداف مدنيين")
            if msg.classification.destroying_homes:
                flags.append("🏠 تدمير منازل")
            if msg.classification.targeting_facilities:
                flags.append("🏥 منشآت")
            if msg.classification.forced_displacement:
                flags.append("🚶 تهجير")
            if msg.classification.blocking_aid:
                flags.append("🚫 مساعدات")
            if msg.classification.systematic_violence:
                flags.append("⚠️ عنف ممنهج")

        return flags

    @staticmethod
    def _categorize_messages(messages: List[Message]) -> Dict[str, List[Message]]:
        categories = defaultdict(list)

        for msg in messages:
            if msg.classification:
                if msg.classification.civilian_deaths:
                    categories["قتل مدنيين"].append(msg)
                if msg.classification.targeting_civilians:
                    categories["استهداف مدنيين"].append(msg)
                if msg.classification.destroying_homes:
                    categories["تدمير منازل"].append(msg)
                if msg.classification.targeting_facilities:
                    categories["استهداف منشآت"].append(msg)
                if msg.classification.forced_displacement:
                    categories["تهجير قسري"].append(msg)
                if msg.classification.blocking_aid:
                    categories["عرقلة مساعدات"].append(msg)
                if msg.classification.systematic_violence:
                    categories["عنف ممنهج"].append(msg)

        return dict(categories)

    @staticmethod
    def format_short_summary(messages: List[Message]) -> str:
        count = len(messages)
        date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

        categories = DigestFormatter._categorize_messages(messages)
        category_summary = ", ".join([f"{k}: {len(v)}" for k, v in categories.items()])

        return f"""🔴 تقرير إبادة جماعية

📅 {date_str}
📊 {count} حادثة
🏷️ {category_summary}

📢 @{messages[0].channel if messages else "unknown"}"""


def main():
    logging.basicConfig(level=logging.INFO)

    # Test with sample data
    from database import Message, Classification
    from datetime import datetime

    messages = [
        Message(
            message_id=1,
            channel="test",
            text="Test message 1",
            date=datetime.utcnow(),
            url="https://t.me/test/1",
            classification=Classification(civilian_deaths=True, is_genocidal=True),
        ),
        Message(
            message_id=2,
            channel="test",
            text="Test message 2 with longer text to see how it handles longer content",
            date=datetime.utcnow(),
            url="https://t.me/test/2",
            classification=Classification(targeting_civilians=True, is_genocidal=True),
        ),
    ]

    digest = DigestFormatter.format_digest(messages, "test_batch_001")
    print(digest)


if __name__ == "__main__":
    main()
