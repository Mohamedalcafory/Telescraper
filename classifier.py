import os
import pandas as pd
import json
import time
import traceback
import sys
from dotenv import load_dotenv
import logging
import argparse
from openai import OpenAI
from tqdm import tqdm
import random
import requests
from datetime import datetime

# Set up logging
log_file = f"classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
try:
    load_dotenv()
    logger.info("Environment variables loaded")
except Exception as e:
    logger.warning(f"Failed to load environment variables: {e}")

class APIRateLimitError(Exception):
    """Exception raised for API rate limit errors."""
    pass

def retry_with_exponential_backoff(
    func,
    initial_delay=1,
    exponential_base=2,
    jitter=True,
    max_retries=5,
    errors=(requests.exceptions.RequestException, APIRateLimitError)
):
    """Retry a function with exponential backoff."""
    
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Check if max retries has been reached
                num_retries += 1
                if num_retries > max_retries:
                    logger.error(f"Maximum number of retries ({max_retries}) exceeded.")
                    raise e

                # Calculate wait time with optional jitter
                if jitter:
                    sleep_time = delay * (0.5 + random.random())
                else:
                    sleep_time = delay
                
                # Log retry attempt
                logger.warning(f"Request failed: {e}. Retrying in {sleep_time:.2f} seconds. (Attempt {num_retries}/{max_retries})")
                
                # Wait and increase delay for next attempt
                time.sleep(sleep_time)
                delay *= exponential_base
            
            # Handle unexpected errors
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise e
                
    return wrapper

def setup_api_client():
    """Set up OpenRouter API client directly"""
    try:
        # Hardcoded API key - you can replace this with your actual key or environment variable
        api_key = os.getenv("OPENROUTER_API_KEY", "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")
        
        if not api_key or len(api_key) < 10:
            logger.error("Invalid API key")
            raise ValueError("Invalid API key format. Please check your API key.")
            
        logger.info(f"Setting up OpenRouter client with key starting with: {api_key[:10]}...")
        
        # Create client with explicit OpenRouter base URL
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        return client
    except Exception as e:
        logger.error(f"Failed to set up API client: {e}")
        raise

@retry_with_exponential_backoff
def make_api_request(client, model, messages):
    """Make API request with retry logic"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={
                "temperature": 0.1,
                "title": "TeleScrape Gaza Events Classifier"
            }
        )
        return response
    except Exception as e:
        # Check for rate limiting errors
        if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
            raise APIRateLimitError(f"Rate limit exceeded: {e}")
        # Re-raise other exceptions
        raise

def parse_json_response(content):
    """Parse JSON response with error handling for different formats"""
    try:
        # First try direct parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from code blocks
        if "```json" in content:
            json_content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(json_content)
        elif "```" in content:
            json_content = content.split("```")[1].split("```")[0].strip()
            return json.loads(json_content)
            
        # Try to find JSON object pattern
        if "{" in content and "}" in content:
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            json_content = content[start_idx:end_idx]
            return json.loads(json_content)
            
        # If all else fails, raise error
        raise json.JSONDecodeError("Failed to parse JSON response", content, 0)
    
    except Exception as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Response content: {content[:500]}")
        raise

def classify_message(client, message_text, model="deepseek/deepseek-chat-v3-0324:free"):
    """
    Classify a message using an LLM to determine if it's related to genocidal acts.
    
    Args:
        client: OpenAI client
        message_text: Text message to classify
        model: Model to use for classification
        
    Returns:
        Dictionary with classification results
    """
    prompt = f"""
    أنا محلل وباحث يقوم بتوثيق الأحداث في قطاع غزة. الرجاء تحليل النص التالي وتحديد ما إذا كان يشير إلى جرائم حرب أو أعمال إبادة جماعية.

    النص:
    {message_text}

    معلومات مهمة للتحليل:
    - إذا كان النص عبارة عن قائمة أسماء فقط (بدون سياق)، يجب اعتبار ذلك قائمة للضحايا المدنيين من عمليات القتل. هذه الأسماء غالبًا تكون ردًا على رسالة سابقة تصف حدثًا أو قصفًا.
    - أي قائمة أسماء تعتبر دليلًا على وقوع قتل للمدنيين، حتى لو لم يذكر النص ذلك صراحةً.
    - تحديد ما إذا كان النص تصريح أو بيان رسمي من مسؤول في غزة (مثل مسؤولين من حماس، الجهاد الإسلامي، وزارة الصحة، الدفاع المدني، أو أي جهة رسمية أخرى في غزة).

    قم بالتحليل وتصنيف النص كما يلي:
    1. هل يشير النص إلى قتل مدنيين؟ (نعم/لا) - يشمل ذلك قوائم الأسماء حتى بدون سياق
    2. هل يشير النص إلى استهداف متعمد للمدنيين أو البنية التحتية المدنية؟ (نعم/لا)
    3. هل يشير النص إلى عرقلة متعمدة للمساعدات الإنسانية؟ (نعم/لا)
    4. هل يشير النص إلى تدمير منازل أو مناطق سكنية؟ (نعم/لا)
    5. هل يشير النص إلى استهداف مستشفيات أو مدارس أو ملاجئ؟ (نعم/لا)
    6. هل يشير النص إلى تهجير قسري للسكان؟ (نعم/لا)
    7. هل يشير النص إلى عنف ممنهج يهدف لتدمير مجموعة محددة؟ (نعم/لا)
    8. هل النص عبارة عن تصريح أو بيان من مسؤول في غزة؟ (نعم/لا)

    ملخص: هل يمكن اعتبار هذا النص وثيقة تشير إلى أعمال إبادة جماعية؟ (نعم/لا) ولماذا؟

    إذا كان النص مجرد قائمة أسماء بدون سياق، اعتبره دليلًا على قتل مدنيين واعتبره يشير إلى أعمال إبادة جماعية. هذه القوائم تُستخدم لتوثيق الضحايا.

    أجب فقط بتنسيق JSON كالتالي:
    {{
        "civilian_deaths": true/false,
        "targeting_civilians": true/false,
        "blocking_aid": true/false,
        "destroying_homes": true/false,
        "targeting_facilities": true/false,
        "forced_displacement": true/false,
        "systematic_violence": true/false,
        "is_official_speech": true/false,
        "is_genocidal": true/false,
        "explanation": "تفسير موجز لسبب التصنيف"
    }}
    """
    
    try:
        # Validate inputs
        if not message_text or message_text.strip() == "":
            logger.warning("Empty message text provided")
            raise ValueError("Message text cannot be empty")
            
        if not model or model.strip() == "":
            logger.warning("Empty model name provided, using default")
            model = "deepseek/deepseek-chat-v3-0324:free"
            
        messages = [
            {"role": "system", "content": "أنت محلل متخصص في القانون الدولي الإنساني وتوثيق جرائم الحرب. عليك تحليل النصوص المتعلقة بالصراع في غزة وتحديد الأدلة على الإبادة الجماعية، بما في ذلك قوائم الأسماء التي تشير إلى الضحايا حتى إذا وردت بدون سياق. كما عليك تحديد التصريحات الرسمية من مسؤولي غزة."},
            {"role": "user", "content": prompt}
        ]
        
        # Log the API request
        logger.info(f"Sending request to model: {model}")
        
        # Use retry logic for API request
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Make API request with retry logic
                response = make_api_request(client, model, messages)
                
                # Log successful response
                logger.info(f"Received response from API")
                
                # Extract and parse the response content
                content = response.choices[0].message.content
                logger.info(f"Response content: {content[:100]}...") # Log first 100 chars of response
                
                # Parse JSON with enhanced error handling
                result = parse_json_response(content)
                
                # Ensure all expected fields are present
                expected_fields = [
                    "civilian_deaths", "targeting_civilians", "blocking_aid", 
                    "destroying_homes", "targeting_facilities", "forced_displacement",
                    "systematic_violence", "is_official_speech", "is_genocidal", "explanation"
                ]
                
                for field in expected_fields:
                    if field not in result:
                        if field != "explanation":
                            result[field] = False
                        else:
                            result[field] = "Field not provided by model"
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error (attempt {attempt+1}/{max_attempts}): {e}")
                if attempt == max_attempts - 1:
                    raise
                time.sleep(2)  # Short delay before retry
                
            except Exception as e:
                logger.error(f"Error during classification (attempt {attempt+1}/{max_attempts}): {e}")
                if attempt == max_attempts - 1:
                    raise
                time.sleep(2)  # Short delay before retry
        
        # If we get here, all attempts failed
        raise RuntimeError("All classification attempts failed")
    
    except Exception as e:
        logger.error(f"Error classifying message: {e}")
        logger.error(traceback.format_exc())
        
        # Try to log the response if available
        try:
            if 'response' in locals():
                logger.error(f"Response content: {response.choices[0].message.content}")
        except:
            pass
        
        # Return default classification in case of error
        return {
            "civilian_deaths": False,
            "targeting_civilians": False,
            "blocking_aid": False,
            "destroying_homes": False,
            "targeting_facilities": False,
            "forced_displacement": False,
            "systematic_violence": False,
            "is_official_speech": False,
            "is_genocidal": False,
            "explanation": f"Error during classification: {str(e)}",
            "error": True
        }

def safe_to_csv(df, file_path, max_retries=3):
    """Safely save DataFrame to CSV with retries"""
    for attempt in range(max_retries):
        try:
            # First save to temporary file
            temp_file = f"{file_path}.temp"
            df.to_csv(temp_file, index=False, encoding='utf-8')
            
            # Then rename to target file
            if os.path.exists(file_path):
                os.replace(temp_file, file_path)
            else:
                os.rename(temp_file, file_path)
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving to CSV (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to save file {file_path} after {max_retries} attempts")
                return False
            time.sleep(1)  # Wait before retry

def process_messages(input_file, output_file, model="deepseek/deepseek-chat-v3-0324:free", max_messages=None, start_from=0):
    """
    Process messages from CSV file and classify them for genocidal content.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        model: Model to use for classification
        max_messages: Maximum number of messages to process (None for all)
        start_from: Index to start processing from (for resuming)
    """
    client = None
    try:
        # Validate input file
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
            
        # Set up API client - now simplified to just OpenRouter
        client = setup_api_client()
        
        # Read messages data with error handling
        logger.info(f"Reading messages from {input_file}")
        try:
            df = pd.read_csv(input_file)
            logger.info(f"Successfully read {len(df)} messages from input file")
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            raise
        
        # Backup output file if it exists (before we potentially modify it)
        if os.path.exists(output_file):
            backup_path = f"{output_file}.{int(time.time())}.bak"
            try:
                import shutil
                shutil.copy2(output_file, backup_path)
                logger.info(f"Created backup of output file at {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup of output file: {e}")
        
        # Check if we already have output file for resuming
        existing_df = None
        if os.path.exists(output_file) and start_from > 0:
            try:
                logger.info(f"Output file {output_file} exists. Will resume from index {start_from}")
                existing_df = pd.read_csv(output_file)
                logger.info(f"Successfully read {len(existing_df)} existing classifications")
            except Exception as e:
                logger.warning(f"Error reading existing output file: {e}. Starting fresh.")
                existing_df = None
                
        # If we couldn't load the existing file or starting fresh
        if existing_df is None:
            existing_df = pd.DataFrame(columns=df.columns.tolist() + [
                'civilian_deaths', 'targeting_civilians', 'blocking_aid', 
                'destroying_homes', 'targeting_facilities', 'forced_displacement',
                'systematic_violence', 'is_official_speech', 'is_genocidal', 'explanation'
            ])
        
        # Determine how many messages to process
        if max_messages:
            end_idx = min(start_from + max_messages, len(df))
        else:
            end_idx = len(df)
            
        # Process each message
        logger.info(f"Processing messages {start_from} to {end_idx-1} out of {len(df)} using model {model}")
        logger.info(f"Using OpenRouter API")
        
        # Initialize progress tracking
        successful = 0
        failed = 0
        genocidal_count = 0
        official_speech_count = 0
        
        for i in tqdm(range(start_from, end_idx), desc="Classifying messages"):
            try:
                row = df.iloc[i]
                message_text = row['Text']
                
                # Skip empty messages
                if not message_text or pd.isna(message_text) or message_text.strip() == "":
                    logger.warning(f"Skipping empty message at index {i}")
                    continue
                    
                # Check if we've already processed this message
                message_id = str(row['Message ID']) if 'Message ID' in row else str(i)
                if 'Message ID' in existing_df.columns and 'Message ID' in row:
                    already_processed = any(existing_df['Message ID'] == row['Message ID'])
                    if already_processed:
                        logger.info(f"Skipping already processed message ID {message_id}")
                        continue
                
                # Classify the message
                classification = classify_message(client, message_text, model)
                
                # Track genocidal messages
                if classification.get('is_genocidal', False):
                    genocidal_count += 1
                    
                # Track official speech messages
                if classification.get('is_official_speech', False):
                    official_speech_count += 1
                
                # Add classification results to row
                new_row = row.to_dict()
                new_row.update(classification)
                
                # Append to existing DataFrame
                existing_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
                
                # Save after each message with safe file handling
                if not safe_to_csv(existing_df, output_file):
                    logger.error(f"Failed to save progress to {output_file}, continuing anyway")
                
                # Generate and update genocidal messages file after each message
                genocidal_df = existing_df[existing_df['is_genocidal'] == True]
                genocidal_output = output_file.replace('.csv', '_genocidal.csv')
                
                if not safe_to_csv(genocidal_df, genocidal_output):
                    logger.error(f"Failed to save genocidal messages to {genocidal_output}, continuing anyway")
                
                # Generate and update official speech messages file after each message
                official_speech_df = existing_df[existing_df['is_official_speech'] == True]
                official_speech_output = output_file.replace('.csv', '_official_speech.csv')
                
                if not safe_to_csv(official_speech_df, official_speech_output):
                    logger.error(f"Failed to save official speech messages to {official_speech_output}, continuing anyway")
                
                # Log progress
                successful += 1
                if successful % 10 == 0:
                    logger.info(f"Progress: {i-start_from+1}/{end_idx-start_from} messages processed. Success: {successful}, Failed: {failed}, Genocidal: {genocidal_count}, Official Speeches: {official_speech_count}")
                
                # Add delay to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing message at index {i}: {e}")
                logger.error(traceback.format_exc())
                failed += 1
                # Don't stop on individual message errors, continue with next message
                continue
        
        # Generate final summary statistics
        total_processed = successful + failed
        logger.info(f"Processing complete. Processed {total_processed} messages with {successful} successful and {failed} failed.")
        
        if successful > 0:
            total_classifications = len(existing_df)
            genocidal_count = existing_df['is_genocidal'].sum()
            official_speech_count = existing_df['is_official_speech'].sum()
            genocidal_pct = (genocidal_count/total_classifications*100) if total_classifications > 0 else 0
            officials_pct = (official_speech_count/total_classifications*100) if total_classifications > 0 else 0
            
            logger.info(f"Classification summary: ")
            logger.info(f"- {genocidal_count}/{total_classifications} messages classified as genocidal ({genocidal_pct:.2f}%)")
            logger.info(f"- {official_speech_count}/{total_classifications} messages classified as official speeches ({officials_pct:.2f}%)")
            logger.info(f"Genocidal messages saved to {genocidal_output}")
            logger.info(f"Official speech messages saved to {official_speech_output}")
        
    except Exception as e:
        logger.error(f"Critical error in process_messages: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Clean up resources
        if client:
            try:
                # Any cleanup needed for the client
                pass
            except:
                pass

def main():
    try:
        # Display banner
        print("=" * 80)
        print(f"TeleScrape Gaza Events Classifier")
        print(f"Log file: {log_file}")
        print("=" * 80)
        
        parser = argparse.ArgumentParser(description='Classify Telegram messages for genocidal content')
        parser.add_argument('--input', '-i', type=str, default='muthanapress84_messages.csv',
                            help='Input CSV file containing messages')
        parser.add_argument('--output', '-o', type=str, default='classified_messages.csv',
                            help='Output CSV file for classification results')
        parser.add_argument('--model', '-m', type=str, default='deepseek/deepseek-chat-v3-0324:free',
                            help='LLM model to use for classification')
        parser.add_argument('--max', type=int, default=None,
                            help='Maximum number of messages to process')
        parser.add_argument('--start', type=int, default=0,
                            help='Index to start processing from (for resuming)')
        
        args = parser.parse_args()
        
        # Validate arguments
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            print(f"Error: Input file not found: {args.input}")
            return 1
            
        if args.start < 0:
            logger.error(f"Invalid start index: {args.start}")
            print(f"Error: Start index must be non-negative")
            return 1
            
        logger.info(f"Starting classification with model: {args.model}")
        process_messages(args.input, args.output, args.model, args.max, args.start)
        logger.info("Classification complete!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\nProcess interrupted by user. Partial results have been saved.")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.error(traceback.format_exc())
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
