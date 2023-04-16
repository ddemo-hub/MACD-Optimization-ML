from src.globals import Globals
from src.singleton import Singleton
from .config_service import ConfigService

import requests
import shutil
import pandas
import os

class DataService(metaclass=Singleton):
    def __init__(self, config_service: ConfigService):
        self.config_service = config_service
        
    def download_historical_candles(self, symbol: str, interval: str):
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
                    print(f"[INFO] URL: '{base_url}/{symbol}-{interval}-{year}-{str(month).zfill(2)}.zip' does not exists")
                elif int(response.status_code % 100) == 5:
                    raise Exception(f"[ERROR] {response.status_code}: Binance Server Error. Try again later")
        
        if len(os.listdir(path=path)) == 0:
            raise Exception(f"[ERROR] Something went wrong. 'symbol' or 'interval' parameters might not be correct")
        
        # Merge the data into one csv file
        columns=["timestamp", "open", "high", "low", "close", "volume"]
        final_df = pandas.DataFrame(columns=columns)
        for csv_path in sorted(os.listdir(path=path)):
            current_file = pandas.read_csv(f"{path}/{csv_path}", header=None, names=columns + ["ct", "qav", "not", "tbbav", "tbqav", "ig"])            
            final_df = pandas.concat([final_df, current_file[columns]], axis=0)
        
        [os.remove(f"{path}/{csv_path_}") for csv_path_ in os.listdir(path=path)]
        
        # Save the final csv
        final_df.sort_values(by="timestamp", ignore_index=True).to_csv(f"{path}/ohlcv.csv", index=False)
        
        
    def read_candles(self, start_ts: int, end_ts: int, interval: int, symbol: str):
        cache_path = f"{Globals.klines_path}/{symbol}/{interval}"
        
        if not os.path.exists(cache_path):
            self.download_historical_candles(symbol=symbol, interval=interval)
        
        ohlcv = pandas.read_csv(f"cache_path/{ohlcv.csv}")
        ohlcv = ohlcv.loc[(ohlcv.timestamp >= start_ts) & (ohlcv.timestamp <= end_ts)]
        
        return ohlcv
