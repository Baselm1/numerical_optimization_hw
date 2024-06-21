########################################
######## DATA HANDLER CLASS ############
########################################

'''
Class Description: 
    This class is responsible for fetching the needed data for testing and optimizing the strategy based on the users needs.
    [Fill in Details]
'''

# Dependecies:

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm
from datetime import *
import requests
import os
import asyncio
import aiohttp
from datetime import datetime, timedelta
import zipfile

# Class:

class DataHandler:

    # Class Constants: 
    YEARS = ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
    INTERVALS = ["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1mo"]
    MONTHS = list(range(1,13))
    BASE_URL = 'https://data.binance.vision/data/spot/monthly/klines' # We only care about KLines data, can be later modified for different trading options.
    START_DATE = date(int(YEARS[0]), MONTHS[0], 1) # Start date is on 01/2017. For most trading pairs we won't find data before 2018-2020.
    END_DATE = datetime.date(datetime.now())
    BASE_DIR = './'
    STABLE_COINS = ['USDT','FDUSD','USDC','TUSD','DAI','AEUR'] # baseAsset must not be one of these, we don't want to trade stable coins.
    QUOTE_ASSETS = ['USDT','FDUSD','USDC','TUSD','DAI','AEUR','BTC','ETH','BNB'] # Valid quoteAssets that can be traded against.

    '''
    # Rate Limiting:
    Binance API Endpoints have request limits as any other service or API.
    For many services, the limits don't matter to us. 
    For general requests, the limit is up to 5 requests per second, and up to 60,000 requests per 5 minutes.
    We'll limit the rates per second using asyncio Semaphores. 
    For the 5 minute limit we simply count how many requests we've sent for every 5 minutes. 
    '''

    RATE_LIMIT_SECOND = 20
    RATE_LIMIT_5_MINUTES = 60000

    SEMAPHORE = asyncio.Semaphore(RATE_LIMIT_SECOND) 
    REQUESTS = [] # Keep track of requests every 5 minutes.

    def __init__(self, interval='1m', quote_asset='USDT', min_volume=1e6, number_of_assets=10, change_tol=20):
        '''
            Constructor Description:
                Builds a DataHandler object.

            Inputs:
                ALL inputs are optional by default.
                - interval: The interval of the candlestick data, set by default to 1 minute data.
                - quote_asset: The assets that we're trading in respect to. Most common quote assets are USDT, BTC and ETH.
                - min_volume: The volume by which we filter the markets we want to fetch. Markets that trade more than a million USDT daily are liquid enough and are good for trading. 
                - number_of_assets: The number of assets to return satisfying quote_asset and min_volume (Up to and including, in some cases we might have less than the given number).
                - change_tol: The percentage of 24 hours change in trading volume, assumed by default that if the asset gained or lost 20% then the trading volume will be high and it might be a one time thing...
                therefore we don't want this asset.
        '''

        # First we check the validity of inputs:
        if not interval in self.INTERVALS:
            raise ValueError("The provided interval is invalid!")
        if not quote_asset in self.QUOTE_ASSETS:
            raise ValueError("The provided quote asset is invalid!")
        
        # Now we define the attributes:
        self.interval = interval
        self.quote_asset = quote_asset
        self.min_volume = min_volume
        self.number_of_assets = number_of_assets
        self.change_tol = change_tol

    def get_trading_pairs(self):
        '''
            Description: 
                Fetches all of the relevant ACTIVELY TRADING pairs from Binance's historical data.
                We send two requests to two different endpoints (since each has different wanted information) and then parse the info into a list of relevant trading pairs. 
            Inputs:
                None
            Outputs:
                - Return a list of all files satisfying the requirements.
        '''
        
        # The API Endpoints for the relative information are: 
        exchange_info_url = "https://api.binance.com/api/v3/exchangeInfo"
        ticker_24hr_url = "https://api.binance.com/api/v3/ticker/24hr"

        # Now we send requests to the server and check if we get the proper responses
        exchange_info_response = requests.get(exchange_info_url)
        ticker_24hr_response = requests.get(ticker_24hr_url)

        if exchange_info_response.status_code==200 and ticker_24hr_response.status_code==200: # Verify that we got an OK status.
            
            # Parse the information as JSON. Returns a useful python dicationary for each.

            exchange_info = exchange_info_response.json()
            ticker_24hr_info = ticker_24hr_response.json()
            
            # We then combine the different results of the two different reponses.

            symbol_info_map = {}
            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info['symbol']
                base_asset = symbol_info['baseAsset']
                quote_asset = symbol_info['quoteAsset']
                symbol_info_map[symbol] = {
                    'symbol': symbol,
                    'baseAsset': base_asset,
                    'quoteAsset': quote_asset
                }

            t = []
            for ticker in ticker_24hr_info:
                symbol = ticker['symbol']
                if symbol in symbol_info_map:
                    combined_info = symbol_info_map[symbol]
                    combined_info['priceChangePercent'] = float(ticker['priceChangePercent'])
                    combined_info['quoteVolume'] = float(ticker['quoteVolume'])
                    t.append(combined_info)
            
            # We filter the pairs based on if they meet the user defined criteria.

            result = []
            for pair in t:
                if pair['quoteAsset'] == self.quote_asset and int(pair['quoteVolume']) >= self.min_volume and np.abs(int(pair['priceChangePercent'])) < self.change_tol and not pair['baseAsset'] in self.STABLE_COINS:
                    result.append(pair)
            
            # Lastly, we sort the list by the volume from highest to lowest, and return the min(len(list), self.number_of_assets)

            pairs_to_return = min(self.number_of_assets, len(result))
            result.sort(key=lambda x: x['quoteVolume'], reverse=True)

            return result[:pairs_to_return]
        else: 
            raise RuntimeError(f'Return status codes: {exchange_info_response.status_code}, {ticker_24hr_response.status_code}')        
    
    async def rate_limiter(self):
        '''
            Description:
                Enforces rate limits by counting how many requests we've sent in the last 5 minutes.
        '''

        while True:
            # Remove requests that are older than 5 minutes            
            now = datetime.now()
            while self.REQUESTS and self.REQUESTS[0] < now - timedelta(minutes=5):
                self.REQUESTS.pop(0)
            # If we've reached the 60000 requests per 5 minutes limit, wait
            if len(self.REQUESTS) >= self.RATE_LIMIT_5_MINUTES:
                await asyncio.sleep(1)
            else:
                break

    async def download_file(self, session, url, folder):
        '''
            Description:
                This function downloads a file in an asynchronous fashion.
                This way we cut down a lot on waiting times between sending and receiving requests.
            
            Inputs:
                - session: aiohttp Session object.
                - url: The URL of the file we wish to download (if the file exists).
                - folder: The directory we wish to write the file to.
            
            Returns:
                None
        '''

        # We first define the filename and the filepath.
        filename = os.path.basename(url)
        filepath = os.path.join(folder, filename)

        # Now we check if the file already exists to save on the download.
        if os.path.exists(filepath):
            tqdm.write(f"Skipping {filename}. File already exists.")
            return
        
        # Otherwise we continue.
        await self.rate_limiter() # Enforce rate limits.
        async with self.SEMAPHORE:
            self.REQUESTS.append(datetime.now())
            try:
                async with session.get(url, ssl=False) as response:
                    if response.status == 200:
                        # Creates the progress bar for the current download
                        content_length = int(response.headers.get('Content-Length', 0))
                        with open(filepath, 'wb') as f:
                            progress_bar = tqdm(total=content_length, desc=filename, unit='B', unit_scale=True, bar_format="{percentage:.0f}% {desc}", ncols=50)
                            async for chunk in response.content.iter_any():
                                f.write(chunk) # Write to file and update progress bar as we go.
                                progress_bar.update(len(chunk))
                            progress_bar.close()
                        tqdm.write(f"\r{' ' * 50}\r", end='') # Clears the progress bar for the current download.
                        tqdm.write(f"{filename} downloaded successfully.")
                    else:
                        tqdm.write(f"Failed to download {filename}. Server returned status {response.status}.")
            except aiohttp.ClientError as e:
                tqdm.write(f"Error downloading {filename}: {str(e)}")

    
    async def download_all(self, url_list):
        '''
            Description:
                This function downloads all files in the list supplied.
                It is done in an asynchronous way to cut down on waiting time.

            Inputs:
                - url_list: A list of the URLs of all files we wish to download.
            
            Returns:
                None
        '''

        # First, we check if the download directory exists, and if not we make it. Assumed './data' directory exists.
        download_dir = os.path.join('data','downloads')
        os.makedirs(download_dir, exist_ok=True) 

        # Now we loop over the URLs and create a task for each download such that we can pass that to the 'gather' function of asyncio.
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in url_list:
                tasks.append(self.download_file(session, url, download_dir)) # Appends coroutines to the list of tasks.
            await asyncio.gather(*tasks) # Await downloads

    def download(self):
        '''
            Description:
                This is the "main" function of the downloading logic.
                We first all the pairs we want to download, then list all the possible links we can to download before we pass the links to the download all function.
            
            Inputs/ Outputs:
                None
        '''
        
        # First, prepare a list of all pairs of interest.
        pairs = self.get_trading_pairs()
        url_list = []
        for pair in pairs:
            for year in self.YEARS:
                for month in self.MONTHS:
                    url = os.path.join(self.BASE_URL, pair['symbol'], self.interval,f'{pair['symbol']}-{self.interval}-{year}-{month:02}.zip') 
                    url_list.append(url)

        asyncio.run(self.download_all(url_list)) # Run the download_all method on the list of URLs we got.

    def process_files(self):
        '''
            Description:
                Processes the downloads into usable data.
                We list here the list of all downloaded files, find the unique pair names (Needed to do after download since some pairs might not download or have non existing files on the server) and 
        '''

        # First we define the header of each .csv file, since by default it's not there, we need to add it.
        # The name of everything after 'close' doesn't matter as we don't need those columns.
        headers = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', '1', '2', '3', '4'] 
        

        # Unzip all of the downloaded files. Working under the assumption that downloads have finished and exist.
        for file in os.listdir(os.path.join('data','downloads')):
            if file.endswith('.zip'): # Ignore other hidden system files like .DS_Store on mac.
                filename = file.split('.')[0] + '.csv' # Create the output file name.
                with zipfile.ZipFile(os.path.join('data','downloads',file), 'r') as zip_ref:
                    zip_ref.extract(filename, os.path.join('data', 'downloads'))
                os.remove(os.path.join('data','downloads',file)) # Remove compressed file.
        
        # Now we loop again but this time to clean up the files and store them in separate directories per pair.
        for file in os.listdir(os.path.join('data','downloads')):
            if file.endswith('.csv'):
                pair = file.split('-')[0]
                if not pair in os.listdir('data'):
                    # If the pair doesn't have a directory for it, we create a new clean one.
                    os.mkdir(os.path.join('data', pair))
                
                '''
                    Cleaning the data is gonna be necessary since we have 7 extra columns we don't need.
                    Additionally, we can store everything as .parquet to save more on space instead of .csv
                '''
                # Read the file:
                df = pd.read_csv(os.path.join('data', 'downloads', file),header=None, names=headers)
                
                # Drop all unwanted columns:
                df.drop(columns=['volume', 'close_time', 'quote_volume', '1', '2', '3', '4'], inplace=True)

                # Save file as parquet to save more on space:
                df.to_parquet(os.path.join('data', pair,file.split('.')[0]+'.parquet'))

                # Delete .csv file as we're done with it:
                os.remove(os.path.join('data', 'downloads', file))

    def run(self):
        # Download all files of interest:
        self.download()
        # Process all the files:
        self.process_files()
        print('Data Download Finished')


handler = DataHandler(min_volume=50e6, number_of_assets=3)

handler.run()