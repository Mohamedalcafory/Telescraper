from telethon import TelegramClient, errors
import asyncio
import pandas as pd
from datetime import datetime, timezone
from tqdm.asyncio import tqdm 
import os
import random
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("telegram_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

api_id = 26874679
api_hash = 'bbdaf41297108352425ea2531fea0147'
my_phone = '+201010622257'
channel_username = 'muthanapress84'

# Create the client and connect using a session file name
# The session file ('my_telegram_session') stores your authorization
# so you don't have to log in every time.
client = TelegramClient('my_telegram_session', api_id, api_hash)

async def authorization(client, phone):
    # Connect to Telegram
    try:
        await client.start(phone)
        logger.info("Client Created and Connected Successfully!")

        # Ensure you're authorized (Telethon handles the login flow if needed)
        if not await client.is_user_authorized():
            logger.info("Client is not authorized. Please follow the prompts to log in.")
            await client.send_code_request(phone)
            try:
                await client.sign_in(phone, input('Enter the code: '))
            except Exception as e:
                logger.error(f"Failed to sign in: {e}")
                return None
        me = await client.get_me()
        logger.info(f"Successfully connected as: {me.first_name} (@{me.username})")
        return client
    except errors.FloodWaitError as e:
        wait_time = e.seconds
        logger.error(f"Hit rate limit during authorization. Need to wait {wait_time} seconds.")
        await asyncio.sleep(wait_time)
        return await authorization(client, phone)
    except Exception as e:
        logger.error(f"Error during authorization: {e}")
        return None

async def retry_with_backoff(coro, max_retries=5, initial_delay=1):
    """Retry an async coroutine with exponential backoff."""
    retries = 0
    delay = initial_delay
    
    while True:
        try:
            return await coro
        except errors.FloodWaitError as e:
            # Handle Telegram's rate limiting
            wait_time = e.seconds
            logger.warning(f"Rate limited. Waiting for {wait_time} seconds as instructed by Telegram")
            await asyncio.sleep(wait_time)
            # Reset retry counter after a flood wait
            retries = 0
            delay = initial_delay
        except (errors.ServerError, errors.TimedOutError, ConnectionError) as e:
            if retries >= max_retries:
                logger.error(f"Maximum retries reached. Last error: {e}")
                raise
            
            # Calculate backoff delay with jitter
            jitter = random.uniform(0.1, 0.5)
            sleep_time = delay + jitter
            
            logger.warning(f"Attempt {retries+1}/{max_retries} failed with error: {e}. Retrying in {sleep_time:.2f} seconds")
            await asyncio.sleep(sleep_time)
            
            # Increase delay for next retry
            retries += 1
            delay *= 2

async def get_message_chunk(client, channel, offset_id=0, limit=100, offset_date=None):
    """Get a chunk of messages with retry logic."""
    return await retry_with_backoff(
        client.get_messages(
            channel, 
            limit=limit,
            offset_id=offset_id,
            offset_date=offset_date
        )
    )

async def scrape_channel(client, channel_username, start_date=None, end_date=None, batch_size=100, max_messages=None, checkpoint_file=None):
    """
    Scrape messages from a Telegram channel within a specified date range.
    
    Args:
        client: Telegram client
        channel_username: Username of the channel to scrape
        start_date: Optional start date to filter messages
        end_date: Optional end date to filter messages
        batch_size: Number of messages to process in each API request
        max_messages: Maximum number of messages to scrape (None for unlimited)
        checkpoint_file: File to store progress for resuming scraping
        
    Returns:
        DataFrame containing all scraped messages
    """
    try:
        # Get the channel entity
        logger.info(f"Accessing channel: @{channel_username}")
        channel = await retry_with_backoff(client.get_entity(channel_username))
        
        # Prepare output file
        output_file = f'{channel_username}_messages.csv'
        
        # Print scraping information
        logger.info(f"Scraping messages from {channel.title}...")
        date_info = []
        if start_date:
            date_info.append(f"since {start_date.strftime('%Y-%m-%d')}")
        if end_date:
            date_info.append(f"before {end_date.strftime('%Y-%m-%d')}")
        if date_info:
            logger.info(f"Time range: {' and '.join(date_info)}")
        else:
            logger.info("Retrieving all available messages")
            
        # Track progress
        message_count = 0
        batch_counter = 0
        messages_data = []
        
        # Set up for resuming if a checkpoint exists
        last_message_id = 0
        if checkpoint_file and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    last_message_id = int(f.read().strip())
                logger.info(f"Resuming scraping from message ID: {last_message_id}")
            except Exception as e:
                logger.error(f"Error reading checkpoint file: {e}")
        
        # Create progress bar without a known total
        pbar = tqdm(desc="Scraping messages", unit="msg")
        
        # Variable to track if we've reached the start_date
        reached_start_date = False
        current_offset_id = 0
        
        while True:
            # Check if we've hit the max_messages limit
            if max_messages is not None and message_count >= max_messages:
                logger.info(f"Reached maximum message limit: {max_messages}")
                break
                
            # Adjust the batch size for the next request
            current_batch_size = min(batch_size, 
                                     max_messages - message_count if max_messages is not None else batch_size)
            
            # Get a batch of messages
            messages = await get_message_chunk(
                client, 
                channel,
                offset_id=current_offset_id,
                limit=current_batch_size,
                offset_date=end_date
            )
            
            # If no messages returned, we've reached the end
            if not messages:
                logger.info("No more messages to retrieve")
                break
                
            # Process each message in the batch
            for message in messages:
                # Skip messages before start_date if specified
                if start_date and message.date < start_date:
                    reached_start_date = True
                    continue
                
                # Skip messages we've already processed (for resuming)
                if message.id <= last_message_id:
                    continue
                    
                # Extract message data
                messages_data.append({
                    'Message ID': message.id,
                    'Date': message.date.strftime('%Y-%m-%d %H:%M:%S'),
                    'Text': message.text or "", 
                    'URL': f"https://t.me/{channel_username}/{message.id}"
                })
                
                message_count += 1
                pbar.update(1)
                
                # Save current position to checkpoint file
                if checkpoint_file:
                    with open(checkpoint_file, 'w') as f:
                        f.write(str(message.id))
                
                # Update the offset_id for the next batch
                current_offset_id = message.id
            
            # If we've reached or passed the start_date, we can stop
            if reached_start_date:
                logger.info("Reached start date boundary. Stopping.")
                break
                
            # Save batch when reaching batch_size
            if len(messages_data) >= batch_size:
                df_batch = pd.DataFrame(messages_data)
                
                # If first batch, create file, otherwise append
                mode = 'w' if batch_counter == 0 else 'a'
                header = batch_counter == 0
                
                df_batch.to_csv(output_file, index=False, mode=mode, header=header)
                logger.info(f"Saved batch {batch_counter+1} ({len(messages_data)} messages)")
                
                batch_counter += 1
                messages_data = []
            
            # Add a delay between batches to avoid hitting rate limits
            await asyncio.sleep(random.uniform(1, 3))
        
        # Save any remaining messages
        if messages_data:
            df_remaining = pd.DataFrame(messages_data)
            mode = 'w' if batch_counter == 0 else 'a'
            header = batch_counter == 0
            
            df_remaining.to_csv(output_file, index=False, mode=mode, header=header)
            logger.info(f"Saved final batch ({len(messages_data)} messages)")
            batch_counter += 1
        
        pbar.close()
        
        # Load and return complete DataFrame
        logger.info(f"Finished scraping. Retrieved {message_count} messages total.")
        logger.info(f"Data saved to: {os.path.abspath(output_file)}")
        
        # Return the complete dataframe
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            return pd.read_csv(output_file)
        else:
            logger.warning("No messages were saved.")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error while scraping: {e}", exc_info=True)
        return pd.DataFrame()


async def main():
    try:
        client_instance = await authorization(client, my_phone)
        if not client_instance:
            logger.error("Authorization failed. Exiting.")
            return
            
        await asyncio.sleep(1)  # Wait a moment to ensure the client is fully connected

        # Set up date range
        start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2025, 5, 31, tzinfo=timezone.utc)
        
        # Set up checkpoint file for resumable scraping
        checkpoint_file = f"{channel_username}_checkpoint.txt"
        
        # Scrape channel
        await scrape_channel(
            client, 
            channel_username, 
            start_date=start_date, 
            end_date=end_date,
            batch_size=100,  # Smaller batch size to reduce likelihood of hitting limits
            max_messages=None,  # Set to a number if you want to limit
            checkpoint_file=checkpoint_file  # For resumable scraping
        )
        
    except ValueError as ve:
        logger.error(f"Value Error: {ve}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

# Run the main function
with client:
    client.loop.run_until_complete(main())

logger.info("Script finished.")
