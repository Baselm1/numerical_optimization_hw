import asyncio
import aiohttp
import csv
import time

async def get_historical_data(symbol, interval, start_time, end_time=None):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_time}'
    if end_time:
        url += f'&endTime={end_time}'
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Failed to retrieve historical data: {response.status} - {await response.text()}")
                return None

async def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

async def main():
    symbol = 'BTCUSDT'
    interval = '1m'
    
    # Start time: January 1, 2017 (Unix timestamp in milliseconds)
    start_time = 1483228800000
    
    # Get current server time in milliseconds for the end time
    end_time = int(time.time() * 1000)
    
    # Retrieve historical data
    historical_data = await get_historical_data(symbol, interval, start_time, end_time)
    
    if historical_data:
        # Save data to CSV file
        await save_to_csv('btcusdt_data.csv', historical_data)

asyncio.run(main())
