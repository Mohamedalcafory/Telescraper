#!/usr/bin/env python3
import os
import sys
import logging
from dotenv import load_dotenv
from openai import OpenAI
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("api_checker")

# Load environment variables
load_dotenv()

def check_openrouter_api_key():
    """Check if the OpenRouter API key is valid and working"""
    print("\n=== OpenRouter API Key Checker ===\n")
    
    # Get API key from environment variables
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY environment variable is not set.")
        print("   Please set it in your .env file or environment variables.")
        return False
    
    if len(api_key) < 10:
        print(f"❌ Error: API key looks invalid (too short): {api_key[:5]}...")
        return False
    
    print(f"📋 Found API key starting with: {api_key[:8]}...")
    
    try:
        # Create OpenRouter client
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/TeleScrape",
                "X-Title": "TeleScrape API Checker"
            }
        )
        
        # Try multiple tests to verify stability
        for i in range(1, 4):
            print(f"\n🧪 Test #{i}: Sending test request to OpenRouter API...")
            
            # Send test request
            response = client.chat.completions.create(
                model="openai/gpt-3.5-turbo:free",
                messages=[{"role": "user", "content": f"Test message #{i}. Reply with 'API key is working'"}],
                max_tokens=20
            )
            
            # Print response
            content = response.choices[0].message.content
            print(f"✅ Test #{i} successful!")
            print(f"📝 Response: {content}")
            
            # Wait a bit between tests
            if i < 3:
                print("⏳ Waiting before next test...")
                time.sleep(2)
        
        print("\n✅ API key is valid and working correctly!")
        print("✅ Successfully completed all test requests.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing API key: {e}")
        
        error_str = str(e).lower()
        if "401" in error_str or "unauthorized" in error_str:
            print("\n❌ Authentication failed (401 Unauthorized)")
            print("   Your API key appears to be invalid, expired, or revoked.")
        elif "rate limit" in error_str:
            print("\n⚠️ Rate limit exceeded")
            print("   Your account may have usage restrictions.")
        
        print("\n📋 Troubleshooting steps:")
        print("1. Check that your API key is correct and not expired")
        print("2. Verify your OpenRouter account is active")
        print("3. Check if you have sufficient credits/quota")
        print("4. Try generating a new API key in your OpenRouter dashboard")
        
        return False

if __name__ == "__main__":
    try:
        check_openrouter_api_key()
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1) 