from src.utils.globals import Globals
from src.utils.singleton import Singleton
from .config_service import ConfigService

from src.utils.logger import Logger

import requests
import shutil
import pandas
import os

class DataService(metaclass=Singleton):
    def __init__(self, config_service: ConfigService):
        self.config_service = config_service
        
    def _download_historical_candles(self, symbol: str, interval: str):
        """Download the kline data for the given interval of the given symbol and merge the data into a single csv file

        Args:
            interval (str): Interval ENUM
            symbol (str): Symbol ENUM

        Raises:
            Exception: If there is a server-side error at Binance endpoint
            Exception: If no data is found to be downloaded, the parameters given may be the cause
        """
        path = f"{Globals.klines_path}/{symbol}/{interval}"
        os.makedirs(path=path)
        
        base_url = f"{self.config_service.binance_data_url}/{symbol}/{interval}"
        
        # For every month of every year in which kline data exists
        years = range(2017, 2024)
        months = range(1, 13)
        for year in years:
            for month in months:
                url = f"{base_url}/{symbol}-{interval}-{year}-{str(month).zfill(2)}.zip"
                response = requests.get(url, stream=True)
                
                # If the data exists, unpack the data
                if response.status_code == 200:
                    with open(f"{path}/{year}-{month}.zip", "wb") as f:
                        f.write(response.content)
                    
                    shutil.unpack_archive(f"{path}/{year}-{month}.zip", path)
                    os.remove(f"{path}/{year}-{month}.zip")
                elif int(response.status_code % 100) == 4:
                    Logger.warn(f"URL: '{base_url}/{symbol}-{interval}-{year}-{str(month).zfill(2)}.zip' does not exists")
                elif int(response.status_code % 100) == 5:
                    error_message = f"{response.status_code}: Binance Server Error. Try again later" 
                    Logger.error(error_message)
                    raise Exception(error_message)
        
        if len(os.listdir(path=path)) == 0:
            error_message = f"Something went wrong. 'symbol' or 'interval' parameters might not be correct"
            Logger.error(error_message)
            raise Exception(error_message)
        
        # Merge the data into one csv file
        columns=["timestamp", "open", "high", "low", "close", "volume"]
        final_df = pandas.DataFrame(columns=columns)
        for csv_path in sorted(os.listdir(path=path)):
            current_file = pandas.read_csv(f"{path}/{csv_path}", header=None, names=columns + ["ct", "qav", "not", "tbbav", "tbqav", "ig"])            
            final_df = pandas.concat([final_df, current_file[columns]], axis=0)
        
        [os.remove(f"{path}/{csv_path_}") for csv_path_ in os.listdir(path=path)]
        
        # Save the final csv
        final_df.sort_values(by="timestamp", ignore_index=True).to_csv(f"{path}/ohlcv.csv", index=False)
        
        
    def read_candles(self, start_ts: int, end_ts: int, interval: str, symbol: str) -> pandas.DataFrame:
        """If the given interval for the given symbol or the given symbol is not cached in the klines folder, 
           the data is downloaded from Binance and the ohlcv data within the the given start_ts-end_ts range is returned

        Args:
            start_ts (int): Start timestamp (inclusive)
            end_ts (int): End timestamp (inclusive)
            interval (str): Interval ENUM
            symbol (str): Symbol ENUM

        Returns:
            pandas.DataFrame: A dataframe with columns ["timestamp", "open", "high", "low"," close", "volume"]
        """
        candles_path = f"{Globals.klines_path}/{symbol}/{interval}"
        
        if not os.path.exists(candles_path):
            Logger.info(f"Requested interval for the requested symbol ({symbol, interval}) does not exists... Downloading data")
            self._download_historical_candles(symbol=symbol, interval=interval)
        
        # Read candles data
        ohlcv = pandas.read_csv(f"{candles_path}/ohlcv.csv")
        
        # Return the candles that are in the requested range
        ohlcv = ohlcv.loc[(ohlcv.timestamp >= start_ts) & (ohlcv.timestamp <= end_ts)]
        ohlcv.reset_index(drop=True, inplace=True)
        
        # Sort the candles by their timestamps in ascending order        
        ohlcv = ohlcv.sort_values(by="timestamp", ascending=True)
        
        return ohlcv
