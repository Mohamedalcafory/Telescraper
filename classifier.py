import os
import pandas as pd
import json
import time
from dotenv import load_dotenv
import logging
import argparse
from openai import OpenAI
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classifier.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

def setup_api_client():
    """Set up OpenRouter API client directly"""
    # Hardcoded API key - you can replace this with your actual key or environment variable
    api_key = os.getenv("OPENROUTER_API_KEY", "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")
    
    logger.info(f"Setting up OpenRouter client with key starting with: {api_key[:10]}...")
    
    # Create client with explicit OpenRouter base URL
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    return client

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

    قم بالتحليل وتصنيف النص كما يلي:
    1. هل يشير النص إلى قتل مدنيين؟ (نعم/لا) - يشمل ذلك قوائم الأسماء حتى بدون سياق
    2. هل يشير النص إلى استهداف متعمد للمدنيين أو البنية التحتية المدنية؟ (نعم/لا)
    3. هل يشير النص إلى عرقلة متعمدة للمساعدات الإنسانية؟ (نعم/لا)
    4. هل يشير النص إلى تدمير منازل أو مناطق سكنية؟ (نعم/لا)
    5. هل يشير النص إلى استهداف مستشفيات أو مدارس أو ملاجئ؟ (نعم/لا)
    6. هل يشير النص إلى تهجير قسري للسكان؟ (نعم/لا)
    7. هل يشير النص إلى عنف ممنهج يهدف لتدمير مجموعة محددة؟ (نعم/لا)

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
        "is_genocidal": true/false,
        "explanation": "تفسير موجز لسبب التصنيف"
    }}
    """
    
    try:
        messages = [
            {"role": "system", "content": "أنت محلل متخصص في القانون الدولي الإنساني وتوثيق جرائم الحرب. عليك تحليل النصوص المتعلقة بالصراع في غزة وتحديد الأدلة على الإبادة الجماعية، بما في ذلك قوائم الأسماء التي تشير إلى الضحايا حتى إذا وردت بدون سياق."},
            {"role": "user", "content": prompt}
        ]
        
        # Log the API request
        logger.info(f"Sending request to model: {model}")
        
        # Use the exact same structure as test.py
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={
                "temperature": 0.1,
                "title": "TeleScrape Gaza Events Classifier"
            }
        )
        
        # Log successful response
        logger.info(f"Received response from API")
        
        # Parse the JSON response
        content = response.choices[0].message.content
        logger.info(f"Response content: {content[:100]}...") # Log first 100 chars of response
        
        # Sometimes the model might return JSON inside a code block
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        result = json.loads(content)
        
        # Ensure all expected fields are present
        expected_fields = [
            "civilian_deaths", "targeting_civilians", "blocking_aid", 
            "destroying_homes", "targeting_facilities", "forced_displacement",
            "systematic_violence", "is_genocidal", "explanation"
        ]
        
        for field in expected_fields:
            if field not in result:
                if field != "explanation":
                    result[field] = False
                else:
                    result[field] = "Field not provided by model"
        
        return result
    
    except Exception as e:
        logger.error(f"Error classifying message: {e}")
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
            "is_genocidal": False,
            "explanation": f"Error during classification: {str(e)}",
            "error": True
        }

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
    try:
        # Set up API client - now simplified to just OpenRouter
        client = setup_api_client()
        
        # Read messages data
        logger.info(f"Reading messages from {input_file}")
        df = pd.read_csv(input_file)
        
        # Check if we already have output file for resuming
        if os.path.exists(output_file) and start_from > 0:
            logger.info(f"Output file {output_file} exists. Will resume from index {start_from}")
            existing_df = pd.read_csv(output_file)
        else:
            existing_df = pd.DataFrame(columns=df.columns.tolist() + [
                'civilian_deaths', 'targeting_civilians', 'blocking_aid', 
                'destroying_homes', 'targeting_facilities', 'forced_displacement',
                'systematic_violence', 'is_genocidal', 'explanation'
            ])
        
        # Determine how many messages to process
        if max_messages:
            end_idx = min(start_from + max_messages, len(df))
        else:
            end_idx = len(df)
            
        # Process each message
        logger.info(f"Processing messages {start_from} to {end_idx-1} out of {len(df)} using model {model}")
        logger.info(f"Using OpenRouter API")
        
        for i in tqdm(range(start_from, end_idx), desc="Classifying messages"):
            row = df.iloc[i]
            message_text = row['Text']
            
            # Skip empty messages
            if not message_text or pd.isna(message_text) or message_text.strip() == "":
                logger.warning(f"Skipping empty message at index {i}")
                continue
                
            # Classify the message
            classification = classify_message(client, message_text, model)
            
            # Add classification results to row
            new_row = row.to_dict()
            new_row.update(classification)
            
            # Append to existing DataFrame
            existing_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save after each message - this is the change the user requested
            existing_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"Saved progress: {i - start_from + 1}/{end_idx - start_from} messages processed")
            
            # Generate and update genocidal messages file after each message
            genocidal_df = existing_df[existing_df['is_genocidal'] == True]
            genocidal_output = output_file.replace('.csv', '_genocidal.csv')
            genocidal_df.to_csv(genocidal_output, index=False, encoding='utf-8')
            
            # Add delay to avoid rate limits
            time.sleep(1)
        
        # Generate final summary statistics
        genocidal_count = existing_df['is_genocidal'].sum()
        total_count = len(existing_df)
        logger.info(f"Classification summary: {genocidal_count}/{total_count} messages classified as genocidal ({genocidal_count/total_count*100:.2f}%)")
        logger.info(f"Genocidal messages saved to {genocidal_output}")
        
    except Exception as e:
        logger.error(f"Error processing messages: {e}")
        raise

def main():
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
    
    logger.info(f"Starting classification with model: {args.model}")
    process_messages(args.input, args.output, args.model, args.max, args.start)
    logger.info("Classification complete!")

if __name__ == "__main__":
    main()
