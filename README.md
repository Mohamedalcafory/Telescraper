# TeleScrape

Telegram pipeline that scrapes messages from a channel, classifies them for genocide-related content, and publishes filtered results to a Telegram bot.

## Overview

```
Scraper → MongoDB → Classifier → Filter → Publisher → Telegram Bot
   ↓           ↓           ↓           ↓          ↓
 Telegram   Storage     AI/LLM     Rules     Delete on success
```

## Features

- **MongoDB-backed storage** - All messages stored in MongoDB with checkpoint sync
- **AI Classification** - Uses OpenRouter API (DeepSeek) to classify messages
- **Filtering** - Filters genocide-related content based on classification
- **Publisher** - Sends formatted messages to Telegram channel
- **Delete on publish** - Removes messages from database after successful publishing
- **Retry queue** - Configurable retries for failed publishes
- **Scheduler** - Runs on configurable intervals (default: 2 hours)
- **Interactive Bot** - Telegram bot for monitoring and managing the pipeline

## Requirements

- Python 3.9+
- MongoDB
- Telegram API credentials
- OpenRouter API key

## Installation

```bash
# Clone and enter directory
cd Telescraper

# Install dependencies
pip install -r requirements.txt

# Copy configuration
cp config/.env.example config/.env
```

## Configuration

Edit `config/.env` with your credentials:

```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=telescraper

# Telegram Scraper
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_PHONE=+1234567890
SOURCE_CHANNEL=muthanapress84

# Telegram Publisher
TELEGRAM_BOT_TOKEN=your_bot_token
PUBLISH_CHANNEL=@your_channel

# OpenRouter
OPENROUTER_API_KEY=your_api_key

# Pipeline Settings
SYNC_INTERVAL_HOURS=2
BATCH_SIZE=50
MAX_RETRY_COUNT=3
RETRY_DELAY_SECONDS=60
CLASSIFICATION_MODEL=deepseek/deepseek-chat-v3-0324:free

# Bot Settings (Optional)
ADMIN_USER_IDS=123456789,987654321
BOT_ENABLE_COMMANDS=true
BOT_ENABLE_CALLBACKS=true
```

## Usage

### Run Full Pipeline Once

```bash
python main.py --run-once
```

### Run on Schedule

```bash
python main.py --schedule
```

### Individual Components

```bash
# Sync messages only
python main.py --sync

# Classify messages only
python main.py --classify

# Publish messages only
python main.py --publish

# Show database statistics
python main.py --stats

# Run the Telegram Bot
python main.py --bot
```

## Bot Commands

### User Commands (Available to All)

| Command | Description |
|---------|-------------|
| `/start` | Start the bot and show welcome message |
| `/help` | Display help information |
| `/latest` | Show latest messages from database |
| `/genocidal` | Show messages classified as genocidal |

### Admin Commands (Requires Admin Access)

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

### Inline Keyboards

The bot provides interactive inline keyboards for:
- Message preview with classification flags
- Publish/Delete actions
- Confirmation dialogs
- Navigation menus

### Running the Bot

```bash
# Run bot in polling mode (default)
python main.py --bot

# Or run directly
python bot/app.py

# Run in webhook mode
python bot/app.py --webhook --url https://your-domain.com --port 8443
```

## Database Schema

### Messages Collection

```javascript
{
  message_id: Number,
  channel: String,
  text: String,
  date: Date,
  url: String,
  classification: {
    civilian_deaths: Boolean,
    targeting_civilians: Boolean,
    blocking_aid: Boolean,
    destroying_homes: Boolean,
    targeting_facilities: Boolean,
    forced_displacement: Boolean,
    systematic_violence: Boolean,
    is_official_speech: Boolean,
    is_genocidal: Boolean,
    explanation: String,
    classified_at: Date,
    model_used: String
  },
  status: String,  // new, classified, pending_publish, published, failed
  retry_count: Number,
  error_message: String,
  published_at: Date,
  created_at: Date,
  updated_at: Date
}
```

### Checkpoints Collection

```javascript
{
  channel: String,
  last_message_id: Number,
  last_sync: Date
}
```

## Message Status Flow

```
new → classified → pending_publish → published → (deleted)
                                      ↓
                                   failed → (retry up to MAX_RETRY_COUNT)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| MONGODB_URI | MongoDB connection string | mongodb://localhost:27017 |
| MONGODB_DATABASE | Database name | telescraper |
| TELEGRAM_API_ID | Telegram API ID | - |
| TELEGRAM_API_HASH | Telegram API hash | - |
| TELEGRAM_PHONE | Phone number | - |
| SOURCE_CHANNEL | Channel to scrape | muthanapress84 |
| TELEGRAM_BOT_TOKEN | Bot token for publishing | - |
| PUBLISH_CHANNEL | Channel to publish to | - |
| OPENROUTER_API_KEY | OpenRouter API key | - |
| SYNC_INTERVAL_HOURS | Sync interval | 2 |
| BATCH_SIZE | Messages per batch | 50 |
| MAX_RETRY_COUNT | Max publish retries | 3 |
| RETRY_DELAY_SECONDS | Delay between retries | 60 |
| ADMIN_USER_IDS | Comma-separated admin user IDs | - |
| BOT_ENABLE_COMMANDS | Enable bot commands | true |
| BOT_ENABLE_CALLBACKS | Enable callback queries | true |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         TeleScrape                              │
├─────────────────────────────────────────────────────────────────┤
│  Scraper → MongoDB → Classifier → Filter → Publisher → Bot    │
│     ↓           ↓           ↓           ↓          ↓          │
│  Telethon   Storage     OpenRouter    Rules    Interactive    │
│                                                       ↓         │
│                                              Telegram Channel   │
└─────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
Telescraper/
├── bot/                    # Telegram bot module
│   ├── app.py             # Bot application
│   ├── handlers.py        # Command handlers
│   ├── keyboards.py       # Inline keyboards
│   ├── services.py       # Business logic
│   ├── config.py         # Bot configuration
│   └── utils.py          # Utilities
├── classifier/           # AI classification
├── config/               # Configuration
├── database/             # MongoDB operations
├── filter/               # Message filtering
├── publisher/            # Telegram publishing
├── scraper/              # Telegram scraping
├── main.py               # Main entry point
└── orchestrator.py       # Pipeline orchestration
```

## License

MIT
