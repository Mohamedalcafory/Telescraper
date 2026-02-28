# TeleScrape Bot Module

Telegram bot module for interactive management and monitoring of the TeleScrape pipeline.

## Overview

The bot module provides an interactive Telegram interface for:
- Monitoring pipeline statistics
- Managing message publishing
- Viewing classified messages
- Manual publishing controls
- Database queries and searches

## Architecture

```
bot/
├── __init__.py         # Module exports
├── app.py              # Main bot application
├── config.py           # Bot configuration
├── handlers.py         # Command and callback handlers
├── keyboards.py        # Inline keyboard layouts
├── services.py         # Business logic layer
└── utils.py            # Utility functions
```

## Bot Commands

### User Commands (Available to All)

| Command | Description |
|---------|-------------|
| `/start` | Start the bot and show welcome message |
| `/help` | Display help information |
| `/latest` | Show latest messages from database |
| `/genocidal` | Show messages classified as genocidal |

### Admin Commands (Restricted)

| Command | Description |
|---------|-------------|
| `/stats` | Show database statistics |
| `/pending` | Show messages pending publication |
| `/failed` | Show failed publication messages |
| `/publish <id>` | Manually publish a specific message |
| `/publish_all` | Publish all pending messages |
| `/retry` | Retry failed messages |
| `/delete <id>` | Delete a message from database |
| `/config` | Show system configuration |
| `/search <query>` | Search messages by text |
| `/admin` | Open admin panel |

### Callback Actions

Interactive buttons provide:
- Message preview with classification flags
- Publish/Delete actions
- Confirmation dialogs
- Navigation menus

## Configuration

Add the following to your `config/.env`:

```env
# Bot Configuration
ADMIN_USER_IDS=123456789,987654321
BOT_ENABLE_COMMANDS=true
BOT_ENABLE_CALLBACKS=true
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ADMIN_USER_IDS` | Comma-separated list of admin Telegram user IDs | Empty (all users allowed) |
| `BOT_ENABLE_COMMANDS` | Enable command handlers | true |
| `BOT_ENABLE_CALLBACKS` | Enable callback queries | true |

## Running the Bot

### Method 1: Via main.py

```bash
python main.py --bot
```

### Method 2: Direct execution

```bash
python bot/app.py
```

### Method 3: Webhook mode

```bash
python bot/app.py --webhook --url https://your-domain.com --port 8443
```

## Database Integration

The bot directly interacts with MongoDB through the existing `database` module:

- **MessageRepository**: CRUD operations on messages
- **CheckpointRepository**: Sync checkpoint management
- **Filter**: Genocidal content filtering
- **Publisher**: Telegram message publishing

## Message Status Flow

```
new → classified → pending_publish → published → (deleted)
                                      ↓
                                   failed → (retry up to MAX_RETRY_COUNT)
```

## Keyboard Layouts

### Main Menu
- Statistics / Latest
- Pending / Failed
- Retry / Help

### Message Actions
- Publish (for pending messages)
- Delete / Get Link
- Back to main menu

### Admin Panel
- Settings / Publish All
- Export / Sync
- Back to main menu

## Error Handling

All errors are:
- Logged to the application log
- Displayed to the user in Arabic
- Handled gracefully with user-friendly messages

## Security

- Admin-only commands require user ID validation
- Confirmation dialogs for destructive actions
- Input sanitization for all user inputs
- MongoDB queries use parameterized operations

## Dependencies

The bot uses:
- `python-telegram-bot` - Telegram Bot API wrapper
- Existing `database` module for data access
- Existing `publisher` module for message publishing

## Extension Points

To add new commands:

1. Add handler method in `handlers.py`
2. Register in `setup_handlers()` function
3. Add keyboard if needed in `keyboards.py`

Example:
```python
async def new_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("New command!")

# In setup_handlers:
application.add_handler(CommandHandler("new", handlers.new_command))
```

## Logging

Logs are written to:
- Console (INFO level)
- Python logging system

Configure via `LOG_LEVEL` in `.env`.

## Troubleshooting

### Bot not responding
1. Check bot token in `.env`
2. Verify MongoDB connection
3. Check admin user IDs if configured

### Commands not working
1. Run `/start` to register commands
2. Check bot has proper permissions
3. Verify admin status for restricted commands

### Database errors
1. Check MongoDB is running
2. Verify `MONGODB_URI` in `.env`
3. Check database permissions

---

**Timestamp**: 2026-02-28 20:50 UTC
**Module Version**: 1.0.0
**TeleScrape Version**: 1.0.0
