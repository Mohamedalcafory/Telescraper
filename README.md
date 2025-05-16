# TeleScrape - Gaza Events Classifier

A tool to document and classify events in Gaza by analyzing Telegram messages for signs of genocidal acts.

## Setup

1. Install dependencies:
   ```
   pip install -e .
   ```

2. Create a `.env` file with your API keys:
   ```
   # Choose one: "openai" or "openrouter"
   API_TYPE=openai

   # OpenAI API Key (required if API_TYPE=openai)
   OPENAI_API_KEY=your_openai_api_key_here

   # OpenRouter API Key (required if API_TYPE=openrouter)
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

## Usage

The project consists of two main components:

### 1. Telegram Scraper (`main.py`)

Scrapes messages from a Telegram channel.

```bash
python main.py
```

### 2. Message Classifier (`classifier.py`)

Analyzes messages for evidence of genocidal acts.

```bash
# Basic usage with default settings
python classifier.py

# With custom options
python classifier.py --input muthanapress84_messages.csv --output classified_messages.csv --model gpt-4-turbo --max 100 --start 0
```

Options:
- `--input`, `-i`: Input CSV file (default: muthanapress84_messages.csv)
- `--output`, `-o`: Output CSV file (default: classified_messages.csv)
- `--model`, `-m`: LLM model to use (default: gpt-4-turbo)
- `--max`: Maximum number of messages to process
- `--start`: Index to start processing from (for resuming)

## Output

The classifier generates two CSV files:
1. `classified_messages.csv` - All messages with classification columns
2. `classified_messages_genocidal.csv` - Only messages classified as genocidal

Classification includes these categories:
- `civilian_deaths`: Indicates civilian casualties
- `targeting_civilians`: Deliberate targeting of civilians
- `blocking_aid`: Blocking humanitarian aid
- `destroying_homes`: Destruction of homes/residential areas
- `targeting_facilities`: Targeting hospitals, schools, or shelters
- `forced_displacement`: Forced displacement of population
- `systematic_violence`: Systematic violence targeting specific groups
- `is_genocidal`: Overall classification as genocidal
- `explanation`: Reasoning for the classification

## License

This project is for humanitarian documentation purposes only.
