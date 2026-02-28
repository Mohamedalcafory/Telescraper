import re
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def escape_markdown(text: str) -> str:
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)


def format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def format_date_arabic(dt: datetime) -> str:
    months_arabic = {
        1: "يناير",
        2: "فبراير",
        3: "مارس",
        4: "أبريل",
        5: "مايو",
        6: "يونيو",
        7: "يوليو",
        8: "أغسطس",
        9: "سبتمبر",
        10: "أكتوبر",
        11: "نوفمبر",
        12: "ديسمبر",
    }
    return f"{dt.day} {months_arabic[dt.month]} {dt.year}"


def truncate_text(text: str, max_length: int = 200) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_stats_display(stats: Dict[str, Any]) -> str:
    lines = ["📊 *إحصائيات قاعدة البيانات*", ""]

    status_map = {
        "new": "📥 جديدة",
        "classified": "📋 مصنفة",
        "pending_publish": "⏳ بانتظار النشر",
        "published": "✅ منشورة",
        "failed": "❌ فاشلة",
    }

    for status, label in status_map.items():
        count = stats.get(status, 0)
        lines.append(f"{label}: {count}")

    lines.append("")
    lines.append(f"📁 *الإجمالي*: {stats.get('total', 0)}")

    return "\n".join(lines)


def format_message_preview(message: Any) -> str:
    text = truncate_text(message.text, 150)

    flags = []
    if message.classification:
        if message.classification.civilian_deaths:
            flags.append("👥 قتل مدنيين")
        if message.classification.targeting_civilians:
            flags.append("🎯 استهداف مدنيين")
        if message.classification.destroying_homes:
            flags.append("🏠 تدمير منازل")
        if message.classification.targeting_facilities:
            flags.append("🏥 منشآت")
        if message.classification.forced_displacement:
            flags.append("🚶 تهجير")
        if message.classification.blocking_aid:
            flags.append("🚫 مساعدات")

    flags_str = " | ".join(flags) if flags else "بدون تصنيف"

    preview = f"""🔖 *الرسالة #{message.message_id}*

{text}

📅 {format_date(message.date)}
🏷️ {flags_str}

[رابط الرسالة]({message.url})"""

    return preview


def format_error_message(error: Exception) -> str:
    error_msg = str(error)
    if len(error_msg) > 500:
        error_msg = error_msg[:497] + "..."
    return f"❌ *خطأ*\n\n{error_msg}"


def parse_callback_data(data: str) -> Dict[str, str]:
    parts = data.split(":")
    if len(parts) >= 2:
        return {
            "action": parts[0],
            "id": parts[1],
            "extra": ":".join(parts[2:]) if len(parts) > 2 else "",
        }
    return {"action": data, "id": "", "extra": ""}


def build_callback_data(action: str, msg_id: str, extra: str = "") -> str:
    if extra:
        return f"{action}:{msg_id}:{extra}"
    return f"{action}:{msg_id}"


def format_help_text() -> str:
    return """🤖 *أوامر البوت*

/start - بدء استخدام البوت
/latest - أحدث الرسائل المصنفة
/stats - إحصائيات قاعدة البيانات
/publish <id> - نشر رسالة محددة
/pending - الرسائل بانتظار النشر
/retry - إعادة المحاولة للرسائل الفاشلة
/help - عرض هذه المساعدة

🔧 *أوامر管理员 (مدير)*

/config - عرض الإعدادات
/publish_all - نشر الكل
/delete <id> - حذف رسالة
/export - تصدير البيانات

💡 *ملاحظات*
- يمكن استخدام الأزرار أسفل الرسائل للتفاعل
- الرسائل المحذوفة لا يمكن استعادتها"""
