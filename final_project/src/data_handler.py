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


# Constants: 

## ADD URLs, Websockets and requests limits.

# Class:

class DataHandler:
    
    def __init__(self) -> None: # Constructor takes in the required parameters for the needed data.
        pass

    async def fetch_data(self): # Async downloads the data from the provided web sockets
        pass

    def process_data(self): # Process the data since the downloaded data might not be formatted or as clean as needed.
        pass

    def run(self): # Run the instance of a data handler as needed.
        pass
