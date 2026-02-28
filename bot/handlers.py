import logging
from typing import Optional
from datetime import datetime, timedelta

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from bot.config import bot_settings
from bot.services import BotService
from bot.keyboards import (
    main_menu_keyboard,
    message_action_keyboard,
    confirm_keyboard,
    admin_keyboard,
    settings_keyboard,
    back_keyboard,
)
from bot.utils import (
    format_stats_display,
    format_message_preview,
    format_error_message,
    format_help_text,
    parse_callback_data,
)

logger = logging.getLogger(__name__)


class BotHandlers:
    def __init__(self):
        self.service = BotService()
        self.user_sessions = {}

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user

        welcome_text = f"""🤖 *مرحباً بك في بوت TeleScrape*

مراقب قنوات Telegram لتصنيف محتوى الإبادة الجماعية.

📊 *الإحصائيات*: {self.service.get_stats().get("total", 0)} رسالة في قاعدة البيانات

اختر من القائمة أدناه أو استخدم الأوامر:
/latest - أحدث الرسائل
/stats - الإحصائيات
/help - المساعدة

━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bot for monitoring Telegram channels and classifying genocide-related content."""

        await update.message.reply_text(
            welcome_text, reply_markup=main_menu_keyboard(), parse_mode="Markdown"
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            format_help_text(), parse_mode="Markdown", reply_markup=back_keyboard()
        )

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not bot_settings.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ ليس لديك صلاحية استخدام هذا الأمر")
            return

        if not self.service.check_database():
            await update.message.reply_text("❌ خطأ في الاتصال بقاعدة البيانات")
            return

        stats = self.service.get_stats()
        await update.message.reply_text(
            format_stats_display(stats), parse_mode="Markdown"
        )

    async def latest_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.service.check_database():
            await update.message.reply_text("❌ خطأ في الاتصال بقاعدة البيانات")
            return

        messages = self.service.get_latest_messages(limit=10)

        if not messages:
            await update.message.reply_text("📭 لا توجد رسائل")
            return

        for msg in messages:
            preview = format_message_preview(msg)
            keyboard = message_action_keyboard(
                msg.message_id,
                msg.status.value if hasattr(msg.status, "value") else str(msg.status),
            )

            await update.message.reply_text(
                preview, parse_mode="Markdown", reply_markup=keyboard
            )

    async def pending_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not bot_settings.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ ليس لديك صلاحية استخدام هذا الأمر")
            return

        if not self.service.check_database():
            await update.message.reply_text("❌ خطأ في الاتصال بقاعدة البيانات")
            return

        messages = self.service.get_pending_messages(limit=10)

        if not messages:
            await update.message.reply_text("📭 لا توجد رسائل معلقة")
            return

        await update.message.reply_text(
            f"📋 *{len(messages)} رسائل معلقة*", parse_mode="Markdown"
        )

        for msg in messages:
            preview = format_message_preview(msg)
            keyboard = message_action_keyboard(
                msg.message_id,
                msg.status.value if hasattr(msg.status, "value") else str(msg.status),
            )

            await update.message.reply_text(
                preview, parse_mode="Markdown", reply_markup=keyboard
            )

    async def failed_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not bot_settings.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ ليس لديك صلاحية استخدام هذا الأمر")
            return

        if not self.service.check_database():
            await update.message.reply_text("❌ خطأ في الاتصال بقاعدة البيانات")
            return

        messages = self.service.get_failed_messages(limit=10)

        if not messages:
            await update.message.reply_text("✅ لا توجد رسائل فاشلة")
            return

        await update.message.reply_text(
            f"❌ *{len(messages)} رسائل فاشلة*", parse_mode="Markdown"
        )

        for msg in messages:
            preview = format_message_preview(msg)
            retry_info = f"\n🔄 محاولات إعادة: {msg.retry_count}"
            if msg.error_message:
                retry_info += f"\n❗️{msg.error_message[:100]}"

            keyboard = message_action_keyboard(msg.message_id, "failed")

            await update.message.reply_text(
                preview + retry_info, parse_mode="Markdown", reply_markup=keyboard
            )

    async def publish_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not bot_settings.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ ليس لديك صلاحية استخدام هذا الأمر")
            return

        if not context.args:
            await update.message.reply_text("⚠️Usage: /publish <message_id>")
            return

        try:
            message_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("⚠️ معرف الرسالة يجب أن يكون رقماً")
            return

        if not self.service.check_database():
            await update.message.reply_text("❌ خطأ في الاتصال بقاعدة البيانات")
            return

        message = self.service.get_message_by_id(
            message_id, bot_settings.SOURCE_CHANNEL
        )

        if not message:
            await update.message.reply_text("❌ الرسالة غير موجودة")
            return

        confirm_keyb = confirm_keyboard(
            f"confirm:publish:{message_id}", f"cancel:{message_id}"
        )

        preview = format_message_preview(message)
        await update.message.reply_text(
            f"هل أنت متأكد من نشر هذه الرسالة؟\n\n{preview}",
            parse_mode="Markdown",
            reply_markup=confirm_keyb,
        )

    async def publish_all_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        if not bot_settings.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ ليس لديك صلاحية استخدام هذا الأمر")
            return

        if not self.service.check_database():
            await update.message.reply_text("❌ خطأ في الاتصال بقاعدة البيانات")
            return

        confirm_keyb = confirm_keyboard("confirm:publish_all", "cancel:publish_all")

        pending_count = len(self.service.get_pending_messages(limit=100))

        await update.message.reply_text(
            f"هل أنت متأكد من نشر {pending_count} رسائل معلقة؟",
            reply_markup=confirm_keyb,
        )

    async def retry_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not bot_settings.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ ليس لديك صلاحية استخدام هذا الأمر")
            return

        if not self.service.check_database():
            await update.message.reply_text("❌ خطأ في الاتصال بقاعدة البيانات")
            return

        count = self.service.retry_failed()

        await update.message.reply_text(f"🔄 تم إعادة محاولة {count} رسائل")

    async def delete_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not bot_settings.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ ليس لديك صلاحية استخدام هذا الأمر")
            return

        if not context.args:
            await update.message.reply_text("⚠️ Usage: /delete <message_id>")
            return

        try:
            message_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("⚠️ معرف الرسالة يجب أن يكون رقماً")
            return

        confirm_keyb = confirm_keyboard(
            f"confirm:delete:{message_id}", f"cancel:delete:{message_id}"
        )

        await update.message.reply_text(
            f"⚠️ هل أنت متأكد من حذف الرسالة #{message_id}؟\n\n❗️لا يمكن التراجع عن هذا الإجراء",
            reply_markup=confirm_keyb,
        )

    async def config_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not bot_settings.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ ليس لديك صلاحية استخدام هذا الأمر")
            return

        config = self.service.get_config_summary()

        config_text = f"""⚙️ *إعدادات النظام*

📱 قناة المصدر: {config["source_channel"]}
📢 قناة النشر: {config["publish_channel"]}
🗄️ قاعدة البيانات: {config["mongodb_database"]}
📦 حجم الدفعة: {config["batch_size"]}
🔄 الحد الأقصى للمحاولات: {config["max_retry"]}
⏰فترة المزامنة: {config["sync_interval_hours"]} ساعة
🤖 نموذج التصنيف: `{config["classification_model"]}`
👥 عدد المشرفين: {config["admin_users"]}"""

        await update.message.reply_text(
            config_text, parse_mode="Markdown", reply_markup=settings_keyboard()
        )

    async def genocidal_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        if not self.service.check_database():
            await update.message.reply_text("❌ خطأ في الاتصال بقاعدة البيانات")
            return

        messages = self.service.get_genocidal_messages(limit=10)

        if not messages:
            await update.message.reply_text("📭 لا توجد رسائل إبادة جماعية")
            return

        count = self.service.get_genocidal_count()
        await update.message.reply_text(
            f"🔴 *{count} رسالة إبادة جماعية*", parse_mode="Markdown"
        )

        for msg in messages:
            preview = format_message_preview(msg)
            keyboard = message_action_keyboard(
                msg.message_id,
                msg.status.value if hasattr(msg.status, "value") else str(msg.status),
            )

            await update.message.reply_text(
                preview, parse_mode="Markdown", reply_markup=keyboard
            )

    async def search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not bot_settings.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ ليس لديك صلاحية استخدام هذا الأمر")
            return

        if not context.args:
            await update.message.reply_text("⚠️ Usage: /search <query>")
            return

        query = " ".join(context.args)

        if not self.service.check_database():
            await update.message.reply_text("❌ خطأ في الاتصال بقاعدة البيانات")
            return

        messages = self.service.search_messages(query, limit=10)

        if not messages:
            await update.message.reply_text(f"📭 لا توجد نتائج للبحث: {query}")
            return

        await update.message.reply_text(
            f"🔍 *نتائج البحث: {len(messages)}*", parse_mode="Markdown"
        )

        for msg in messages:
            preview = format_message_preview(msg)
            keyboard = message_action_keyboard(
                msg.message_id,
                msg.status.value if hasattr(msg.status, "value") else str(msg.status),
            )

            await update.message.reply_text(
                preview, parse_mode="Markdown", reply_markup=keyboard
            )

    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not bot_settings.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ ليس لديك صلاحية استخدام هذا الأمر")
            return

        await update.message.reply_text(
            "⚙️ *لوحة الإدارة*", parse_mode="Markdown", reply_markup=admin_keyboard()
        )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        data = query.data
        user_id = update.effective_user.id

        if not bot_settings.is_admin(user_id):
            await query.edit_message_text("⛔ ليس لديك صلاحية")
            return

        parsed = parse_callback_data(data)
        action = parsed.get("action", "")
        msg_id = parsed.get("id", "")

        if action == "menu":
            await self.handle_menu_callback(query, parsed.get("id", ""))

        elif action == "publish":
            await self.handle_publish_callback(query, msg_id)

        elif action == "confirm":
            await self.handle_confirm_callback(query, parsed.get("extra", ""), msg_id)

        elif action == "cancel":
            await query.edit_message_text("❌ تم الإلغاء")

        elif action == "delete":
            await self.handle_delete_callback(query, msg_id)

        elif action == "admin":
            await self.handle_admin_callback(query, parsed.get("id", ""))

        elif action == "link":
            await self.handle_link_callback(query, msg_id)

        else:
            await query.edit_message_text(f"Unknown action: {action}")

    async def handle_menu_callback(self, query, menu_id: str):
        if menu_id == "stats":
            if not self.service.check_database():
                await query.edit_message_text("❌ خطأ في الاتصال بقاعدة البيانات")
                return
            stats = self.service.get_stats()
            await query.edit_message_text(
                format_stats_display(stats), parse_mode="Markdown"
            )

        elif menu_id == "latest":
            messages = self.service.get_latest_messages(limit=5)
            if not messages:
                await query.edit_message_text("📭 لا توجد رسائل")
                return

            for msg in messages:
                preview = format_message_preview(msg)
                keyboard = message_action_keyboard(
                    msg.message_id,
                    msg.status.value
                    if hasattr(msg.status, "value")
                    else str(msg.status),
                )
                await query.message.reply_text(
                    preview, parse_mode="Markdown", reply_markup=keyboard
                )

            await query.message.reply_text(
                "✅ تم عرض الرسائل", reply_markup=back_keyboard()
            )

        elif menu_id == "pending":
            messages = self.service.get_pending_messages(limit=5)
            if not messages:
                await query.edit_message_text("📭 لا توجد رسائل معلقة")
                return

            for msg in messages:
                preview = format_message_preview(msg)
                keyboard = message_action_keyboard(
                    msg.message_id,
                    msg.status.value
                    if hasattr(msg.status, "value")
                    else str(msg.status),
                )
                await query.message.reply_text(
                    preview, parse_mode="Markdown", reply_markup=keyboard
                )

            await query.message.reply_text(
                "✅ تم عرض الرسائل", reply_markup=back_keyboard()
            )

        elif menu_id == "failed":
            messages = self.service.get_failed_messages(limit=5)
            if not messages:
                await query.edit_message_text("✅ لا توجد رسائل فاشلة")
                return

            for msg in messages:
                preview = format_message_preview(msg)
                keyboard = message_action_keyboard(msg.message_id, "failed")
                await query.message.reply_text(
                    preview, parse_mode="Markdown", reply_markup=keyboard
                )

            await query.message.reply_text(
                "✅ تم عرض الرسائل", reply_markup=back_keyboard()
            )

        elif menu_id == "retry":
            count = self.service.retry_failed()
            await query.edit_message_text(f"🔄 تم إعادة محاولة {count} رسائل")

        elif menu_id == "help":
            await query.edit_message_text(
                format_help_text(), parse_mode="Markdown", reply_markup=back_keyboard()
            )

        elif menu_id == "back":
            await query.edit_message_text(
                "🤖 *القائمة الرئيسية*",
                parse_mode="Markdown",
                reply_markup=main_menu_keyboard(),
            )

    async def handle_publish_callback(self, query, message_id: str):
        try:
            msg_id = int(message_id)
        except ValueError:
            await query.edit_message_text("❌ معرف رسالة غير صالح")
            return

        message = self.service.get_message_by_id(msg_id, bot_settings.SOURCE_CHANNEL)

        if not message:
            await query.edit_message_text("❌ الرسالة غير موجودة")
            return

        success = self.service.publish_message(message)

        if success:
            await query.edit_message_text(f"✅ تم نشر الرسالة #{msg_id}")
        else:
            await query.edit_message_text(f"❌ فشل نشر الرسالة #{msg_id}")

    async def handle_confirm_callback(self, query, action: str, message_id: str):
        parts = action.split(":")

        if parts[0] == "publish":
            if parts[1] == "all":
                count = self.service.publish_all_pending()
                await query.edit_message_text(f"✅ تم نشر {count} رسائل")
            else:
                try:
                    msg_id = int(parts[1])
                except ValueError:
                    await query.edit_message_text("❌ معرف رسالة غير صالح")
                    return

                message = self.service.get_message_by_id(
                    msg_id, bot_settings.SOURCE_CHANNEL
                )
                if message:
                    success = self.service.publish_message(message)
                    if success:
                        await query.edit_message_text(f"✅ تم نشر الرسالة #{msg_id}")
                    else:
                        await query.edit_message_text(f"❌ فشل نشر الرسالة #{msg_id}")

        elif parts[0] == "delete":
            try:
                msg_id = int(parts[1])
            except ValueError:
                await query.edit_message_text("❌ معرف رسالة غير صالح")
                return

            success = self.service.delete_message(msg_id, bot_settings.SOURCE_CHANNEL)

            if success:
                await query.edit_message_text(f"✅ تم حذف الرسالة #{msg_id}")
            else:
                await query.edit_message_text(f"❌ فشل حذف الرسالة #{msg_id}")

    async def handle_delete_callback(self, query, message_id: str):
        try:
            msg_id = int(message_id)
        except ValueError:
            await query.edit_message_text("❌ معرف رسالة غير صالح")
            return

        success = self.service.delete_message(msg_id, bot_settings.SOURCE_CHANNEL)

        if success:
            await query.edit_message_text(f"✅ تم حذف الرسالة #{msg_id}")
        else:
            await query.edit_message_text(f"❌ فشل حذف الرسالة #{msg_id}")

    async def handle_admin_callback(self, query, admin_action: str):
        if admin_action == "config":
            config = self.service.get_config_summary()
            config_text = f"""⚙️ *الإعدادات*

📱 المصدر: {config["source_channel"]}
📢 النشر: {config["publish_channel"]}
🗄️ قاعدة البيانات: {config["mongodb_database"]}
📦 الدفعة: {config["batch_size"]}"""
            await query.edit_message_text(
                config_text, parse_mode="Markdown", reply_markup=back_keyboard()
            )

        elif admin_action == "publish_all":
            count = self.service.publish_all_pending()
            await query.edit_message_text(f"✅ تم نشر {count} رسائل")

    async def handle_link_callback(self, query, message_id: str):
        try:
            msg_id = int(message_id)
        except ValueError:
            await query.edit_message_text("❌ معرف رسالة غير صالح")
            return

        message = self.service.get_message_by_id(msg_id, bot_settings.SOURCE_CHANNEL)

        if message and message.url:
            await query.edit_message_text(
                f"🔗 [رابط الرسالة]({message.url})", parse_mode="Markdown"
            )
        else:
            await query.edit_message_text("❌ الرابط غير متوفر")

    async def handle_error(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.error(f"Update {update} caused error {context.error}")

        error_text = format_error_message(context.error)

        if update.message:
            await update.message.reply_text(error_text, parse_mode="Markdown")
        elif update.callback_query:
            await update.callback_query.edit_message_text(
                error_text, parse_mode="Markdown"
            )


def setup_handlers(application: Application):
    handlers = BotHandlers()

    application.add_handler(CommandHandler("start", handlers.start_command))
    application.add_handler(CommandHandler("help", handlers.help_command))
    application.add_handler(CommandHandler("stats", handlers.stats_command))
    application.add_handler(CommandHandler("latest", handlers.latest_command))
    application.add_handler(CommandHandler("pending", handlers.pending_command))
    application.add_handler(CommandHandler("failed", handlers.failed_command))
    application.add_handler(CommandHandler("publish", handlers.publish_command))
    application.add_handler(CommandHandler("publish_all", handlers.publish_all_command))
    application.add_handler(CommandHandler("retry", handlers.retry_command))
    application.add_handler(CommandHandler("delete", handlers.delete_command))
    application.add_handler(CommandHandler("config", handlers.config_command))
    application.add_handler(CommandHandler("genocidal", handlers.genocidal_command))
    application.add_handler(CommandHandler("search", handlers.search_command))
    application.add_handler(CommandHandler("admin", handlers.admin_command))

    application.add_handler(CallbackQueryHandler(handlers.handle_callback))

    application.add_error_handler(handlers.handle_error)

    return handlers
