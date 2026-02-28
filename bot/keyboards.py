from typing import List, Optional
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from bot.utils import build_callback_data


def main_menu_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("📊 الإحصائيات", callback_data="menu:stats"),
            InlineKeyboardButton("� latest", callback_data="menu:latest"),
        ],
        [
            InlineKeyboardButton("⏳ المعلقة", callback_data="menu:pending"),
            InlineKeyboardButton("🔄 الفاشلة", callback_data="menu:failed"),
        ],
        [
            InlineKeyboardButton("🔁 إعادة المحاولة", callback_data="menu:retry"),
            InlineKeyboardButton("📋 المساعدة", callback_data="menu:help"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def message_action_keyboard(message_id: int, status: str) -> InlineKeyboardMarkup:
    keyboard = []

    if status in ["new", "classified", "pending_publish"]:
        keyboard.append(
            [
                InlineKeyboardButton(
                    "📤 نشر",
                    callback_data=build_callback_data("publish", str(message_id)),
                )
            ]
        )

    keyboard.append(
        [
            InlineKeyboardButton(
                "❌ حذف", callback_data=build_callback_data("delete", str(message_id))
            ),
            InlineKeyboardButton(
                "🔗 الرابط", callback_data=build_callback_data("link", str(message_id))
            ),
        ]
    )

    keyboard.append(
        [InlineKeyboardButton("🔙 القائمة الرئيسية", callback_data="menu:back")]
    )

    return InlineKeyboardMarkup(keyboard)


def confirm_keyboard(action: str, message_id: int) -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(
                "✅ نعم",
                callback_data=build_callback_data("confirm", str(message_id), action),
            ),
            InlineKeyboardButton(
                "❌ لا", callback_data=build_callback_data("cancel", str(message_id))
            ),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def pagination_keyboard(
    current_page: int, total_pages: int, action: str
) -> InlineKeyboardMarkup:
    keyboard = []

    nav_buttons = []
    if current_page > 1:
        nav_buttons.append(
            InlineKeyboardButton(
                "⬅️ السابق", callback_data=f"page:{action}:{current_page - 1}"
            )
        )
    if current_page < total_pages:
        nav_buttons.append(
            InlineKeyboardButton(
                "التالي ➡️", callback_data=f"page:{action}:{current_page + 1}"
            )
        )

    if nav_buttons:
        keyboard.append(nav_buttons)

    keyboard.append(
        [InlineKeyboardButton("🔙 القائمة الرئيسية", callback_data="menu:back")]
    )

    return InlineKeyboardMarkup(keyboard)


def admin_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("⚙️ الإعدادات", callback_data="admin:config"),
            InlineKeyboardButton("📤 نشر الكل", callback_data="admin:publish_all"),
        ],
        [
            InlineKeyboardButton("📊 سجل البيانات", callback_data="admin:export"),
            InlineKeyboardButton("🔄 مزامنة", callback_data="admin:sync"),
        ],
        [InlineKeyboardButton("🔙 القائمة الرئيسية", callback_data="menu:back")],
    ]
    return InlineKeyboardMarkup(keyboard)


def back_keyboard(callback_data: str = "menu:back") -> InlineKeyboardMarkup:
    keyboard = [[InlineKeyboardButton("🔙 رجوع", callback_data=callback_data)]]
    return InlineKeyboardMarkup(keyboard)


def settings_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(
                "📊 إحصائيات قاعدة البيانات", callback_data="settings:stats"
            ),
            InlineKeyboardButton("🔗 القنوات", callback_data="settings:channels"),
        ],
        [
            InlineKeyboardButton("⏰ جدول المزامنة", callback_data="settings:schedule"),
            InlineKeyboardButton("📋 السجلات", callback_data="settings:logs"),
        ],
        [InlineKeyboardButton("🔙 القائمة الرئيسية", callback_data="menu:back")],
    ]
    return InlineKeyboardMarkup(keyboard)


def yes_no_keyboard(
    yes_callback: str, no_callback: str = "menu:back"
) -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("✅ نعم", callback_data=yes_callback),
            InlineKeyboardButton("❌ لا", callback_data=no_callback),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)
