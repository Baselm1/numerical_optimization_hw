########################################
######## DATA HANDLER CLASS ############
########################################

'''
Class Description: 
    This class is responsible for fetching the needed data for testing and optimizing the strategy based on the users needs.
    [Fill in Details]
'''

# Dependecies:

import pandas as pd
import asyncio
import aiohttp
import logging
from tqdm.asyncio import tqdm
import aiofiles
import os
from datetime import *
import requests

# Class:

class DataHandler:

    # Class Constants: 
    YEARS = ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
    INTERVALS = ["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1mo"]
    MONTHS = list(range(1,13))
    BASE_URL = 'https://data.binance.vision/spot/monthly/klines' # We only care about KLines data, can be later modified for different trading options.
    START_DATE = date(int(YEARS[0]), MONTHS[0], 1) # Start date is on 01/2017. For most trading pairs we won't find data before 2018-2020.
    END_DATE = datetime.date(datetime.now())
    API_ENDPOINT = 'https://api.binance.com/api/v3/' # Used to fetch all trading assets and 
    BASE_DIR = './'
    STABLE_COINS = ['USDT','FDUSD','USDC','TUSD','DAI','AEUR'] # baseAsset must not be one of these, we don't want to trade stable coins.
    QUOTE_ASSETS = ['USDT','FDUSD','USDC','TUSD','DAI','AEUR','BTC','ETH','BNB'] # Valid quoteAssets that can be traded against.

    def __init__(self, interval='1m', quote_asset='USDT', min_volume=1e6, number_of_assets=10, change_tol=0.2):
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
            Inputs:

            Outputs:

        '''
        
        response = requests.get(self.ASSETS)
        if response.status_code == 200: # OK HTTP status.
            exchange_info = response.json()
        
    async def download_file(self, file_url):
        pass

    def fetch_data(self): # Async downloads the data from the provided web sockets
        pass

    def process_data(self): # Process the data since the downloaded data might not be formatted or as clean as needed.
        pass

    def run(self): # Run the instance of a data handler as needed.
        pass
