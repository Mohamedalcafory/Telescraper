from telethon import TelegramClient
import asyncio
import pandas as pd
from datetime import datetime, timezone
from tqdm.asyncio import tqdm 
import os

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
    await client.start(phone)
    print("Client Created and Connected Successfully!")

    # Ensure you're authorized (Telethon handles the login flow if needed)
    if not await client.is_user_authorized():
        print("Client is not authorized. Please follow the prompts to log in.")
        # If running for the first time, Telegram will send a code.
        # You might need to manually enter it in the console if running interactively.
        # For non-interactive setups, more advanced handling is needed.
        await client.send_code_request(phone)
        try:
            await client.sign_in(phone, input('Enter the code: '))
        except Exception as e:
            print(f"Failed to sign in: {e}")
            return None
    me = await client.get_me()
    print(f"Successfully connected as: {me.first_name} (@{me.username})")
    return client

async def scrape_channel(client, channel_username, start_date=None, end_date=None, batch_size=1000, max_messages=None):
    """
    Scrape messages from a Telegram channel within a specified date range.
    
    Args:
        client: Telegram client
        channel_username: Username of the channel to scrape
        start_date: Optional start date to filter messages
        end_date: Optional end date to filter messages
        batch_size: Number of messages to save in each batch
        max_messages: Maximum number of messages to scrape (None for unlimited)
        
    Returns:
        DataFrame containing all scraped messages
    """
    try:
        # Get the channel entity
        print(f"Accessing channel: @{channel_username}")
        channel = await client.get_entity(channel_username)
        
        # Prepare output file
        output_file = f'{channel_username}_messages.csv'
        
        # Print scraping information
        print(f"Scraping messages from {channel.title}...")
        date_info = []
        if start_date:
            date_info.append(f"since {start_date.strftime('%Y-%m-%d')}")
        if end_date:
            date_info.append(f"before {end_date.strftime('%Y-%m-%d')}")
        if date_info:
            print(f"Time range: {' and '.join(date_info)}")
        else:
            print("Retrieving all available messages")
            
        # Track progress
        message_count = 0
        batch_counter = 0
        messages_data = []
        
        # Create progress bar without a known total
        pbar = tqdm(desc="Scraping messages", unit="msg")
        
        # Iterate through messages
        async for message in client.iter_messages(
            channel, 
            offset_date=end_date,
            limit=max_messages
        ):
            # Skip messages before start_date if specified
            if start_date and message.date < start_date:
                continue
                
            # Extract message data (with more fields)
            messages_data.append({
                'Message ID': message.id,
                'Date': message.date.strftime('%Y-%m-%d %H:%M:%S'),
                'Text': message.text or "", 
                'URL': f"https://t.me/{channel_username}/{message.id}"
            })
            
            message_count += 1
            pbar.update(1)
            
            # Save batch when reaching batch_size
            if len(messages_data) >= batch_size:
                df_batch = pd.DataFrame(messages_data)
                
                # If first batch, create file, otherwise append
                mode = 'w' if batch_counter == 0 else 'a'
                header = batch_counter == 0
                
                df_batch.to_csv(output_file, index=False, mode=mode, header=header)
                print(f"\nSaved batch {batch_counter+1} ({len(messages_data)} messages)")
                
                batch_counter += 1
                messages_data = []
        
        # Save any remaining messages
        if messages_data:
            df_remaining = pd.DataFrame(messages_data)
            mode = 'w' if batch_counter == 0 else 'a'
            header = batch_counter == 0
            
            df_remaining.to_csv(output_file, index=False, mode=mode, header=header)
            print(f"\nSaved final batch ({len(messages_data)} messages)")
            batch_counter += 1
        
        pbar.close()
        
        # Load and return complete DataFrame
        print(f"\nFinished scraping. Retrieved {message_count} messages total.")
        print(f"Data saved to: {os.path.abspath(output_file)}")
        
        # Return the complete dataframe
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            return pd.read_csv(output_file)
        else:
            print("Warning: No messages were saved.")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error while scraping: {e}")
        return pd.DataFrame()


async def main():
    try:
        client_instance = await authorization(client, my_phone)
        if not client_instance:
            print("Authorization failed. Exiting.")
            return
            
        await asyncio.sleep(1)  # Wait a moment to ensure the client is fully connected

        # Set up date range
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2023, 3, 31, tzinfo=timezone.utc)
        
        # Scrape channel
        await scrape_channel(
            client, 
            channel_username, 
            start_date=start_date, 
            end_date=end_date,
            batch_size=1000,
            max_messages=None  # Set to a number if you want to limit
        )
        
    except ValueError as ve:
        print(f"Value Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the main function
with client:
    client.loop.run_until_complete(main())

print("Script finished.")
