from typing import List
from pathlib import Path
import json
import numpy as np
import pandas as pd
from binance.client import Client

mapping = {
    '1T': 'Client.KLINE_INTERVAL_1MINUTE',
    '3T': 'Client.KLINE_INTERVAL_3MINUTE',
    '5T': 'Client.KLINE_INTERVAL_5MINUTE',
    '15T': 'Client.KLINE_INTERVAL_15MINUTE',
    '30T': 'Client.KLINE_INTERVAL_30MINUTE',
    '1h': 'Client.KLINE_INTERVAL_1HOUR',
    '2h': 'Client.KLINE_INTERVAL_2HOUR',
    '4h': 'Client.KLINE_INTERVAL_4HOUR',
    '6h': 'Client.KLINE_INTERVAL_6HOUR',
    '8h': 'Client.KLINE_INTERVAL_8HOUR',
    '12h': 'Client.KLINE_INTERVAL_12HOUR',
    '1d': 'Client.KLINE_INTERVAL_1DAY',
    '3d': 'Client.KLINE_INTERVAL_3DAY',
    '1w': 'Client.KLINE_INTERVAL_1WEEK',
    '1M': 'Client.KLINE_INTERVAL_1MONTH'
}


class DataManager:
    # CONSTANTS
    KLINES_COLUMNS = ["open_time", "open", "high", "low", "close",
                      "volume", "close_time", "quote_volume", "nb_trades",
                      "taker_buy_volume", "taker_buy_quote_volume"]
    METADATA_FILENAME = "metadata.json"
    DATA_FILENAME = "data.parquet"
    DATASET_FOLDER_PREFIX = "dataset"
    COIN_INDEX_NAME = "coin"
    FEATURE_INDEX_NAME = "feature"

    def __init__(self,
                 database_path: str,
                 quote_asset: str,
                 selected_symbols: List[str],
                 selected_features: List[str],
                 date_start: str = "2016-01-01",
                 date_end: str = "2018-01-01",
                 freq: str = "30T"):
        """
        KLINES FORMAT:

            [
                [
                    1499040000000,      # Open time
                    "0.01634790",       # Open
                    "0.80000000",       # High
                    "0.01575800",       # Low
                    "0.01577100",       # Close
                    "148976.11427815",  # Volume
                    1499644799999,      # Close time
                    "2434.19055334",    # Quote asset volume
                    308,                # Number of trades
                    "1756.87402397",    # Taker buy base asset volume
                    "28.46694368",      # Taker buy quote asset volume
                    "17928899.62484339" # Can be ignored
                ]
            ]

        """
        for s in selected_symbols:
            if not (s.startswith(quote_asset) or s.endswith(quote_asset)):
                raise RuntimeError(f"Error for symbol {s}, for quote asset {quote_asset}")

        self.database_path = database_path
        self.client = Client()
        info = self.client.get_exchange_info()
        symbols = [s["symbol"] for s in info["symbols"] if quote_asset in s["symbol"]]
        self._validate_inputs(selected_symbols, symbols, selected_features)
        self.symbols = selected_symbols
        self.quote_asset = quote_asset
        self.freq = freq
        self.data = self._get_data(selected_symbols, selected_features, date_start, date_end, freq)

    def get_data(self):
        return self.data

    def _validate_inputs(self,
                         selected_symbols: List[str],
                         available_symbols: List[str],
                         selected_features: List[str]):

        if not set(selected_symbols).issubset(set(available_symbols)):
            raise RuntimeError(f"Some selected symbols are not in the available symbols \nSelected symbols : "
                               f"\n{selected_symbols}. \nAvailables symbols: \n{available_symbols}")
        if not set(selected_features).issubset(set(self.KLINES_COLUMNS)):
            raise RuntimeError(f"Some selected features are not in the available features \nSelected features : "
                               f"\n{selected_features}. \nAvailables features: \n{self.KLINES_COLUMNS}")

    def _get_kline_df(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:

        start_date = str(pd.to_datetime(start_date))
        end_date = str(pd.to_datetime(end_date))
        klines = self.client.get_historical_klines(symbol, eval(mapping[self.freq]), start_date, end_date)
        klines_data = np.array(klines)[:, :-1]  # we drop the last column as it says it is useless
        df = pd.DataFrame(data=klines_data, columns=self.KLINES_COLUMNS)
        float_cols = [c for c in df.columns if "time" not in c]
        df[float_cols] = df[float_cols].apply(pd.to_numeric)
        if symbol.startswith(self.quote_asset):
            # reverse the prices if the base is the quote
            df[float_cols] = 1.0 / df[float_cols]
        df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
        df = df.set_index("close_time")
        df = df.ffill()
        return df

    def _get_assets_df(self,
                       selected_symbols: List[str],
                       selected_features: List[str],
                       start_date: str,
                       end_date: str,
                       freq: str) -> pd.DataFrame:

        start_date = str(pd.to_datetime(start_date))
        end_date = str(pd.to_datetime(end_date))

        # We add the freq and remove 1ms to match binance close timeindexes
        time_index = pd.date_range(start=start_date, end=end_date, freq=freq) + pd.Timedelta(freq) - pd.Timedelta('1ms')
        print("time_index", time_index)
        multi_index = pd.MultiIndex.from_product([selected_symbols, selected_features],
                                                 names=[self.COIN_INDEX_NAME, self.FEATURE_INDEX_NAME])
        panel = pd.DataFrame(index=time_index, columns=multi_index)
        for coin in selected_symbols:
            coin_df = self._get_kline_df(coin, start_date, end_date)
            panel[coin] = coin_df[selected_features]
        actual_symbols = []
        for coin in selected_symbols:
            if coin.startswith(self.quote_asset):
                reversed_coin = coin[len(self.quote_asset):] + self.quote_asset
                actual_symbols.append(reversed_coin)
            else:
                actual_symbols.append(coin)
        panel.columns.set_levels(actual_symbols, level=0)
        unavailable_dates = panel[panel.isna().sum(axis=1) >= 1].index
        print(
            f"Unavailable dates (length of {len(unavailable_dates)}/{len(time_index)} timestamps): {unavailable_dates}")
        panel = panel.ffill()
        return panel

    def _find_dataset(self,
                      selected_symbols: List[str],
                      selected_features: List[str],
                      start_date: str,
                      end_date: str,
                      freq: str):
        """return NaN if not found, else the DF"""

        metadata = {
            "quote_asset": self.quote_asset,
            "selected_symbols": selected_symbols,
            "selected_features": selected_features,
            "start_date": start_date,
            "end_date": end_date,
            "freq": freq,
        }

        path = Path(self.database_path)
        for p in list(path.glob("*")):
            meta_path = p / self.METADATA_FILENAME
            with open(meta_path) as json_file:
                found_metadata = json.load(json_file)
            if metadata == found_metadata:
                df = pd.read_parquet(p / self.DATA_FILENAME)
                return df
        return None

    def _get_data(self,
                  selected_symbols: List[str],
                  selected_features: List[str],
                  start_date: str,
                  end_date: str,
                  freq: str) -> pd.DataFrame:

        start_date = str(pd.to_datetime(start_date))
        end_date = str(pd.to_datetime(end_date))

        # FIND DF
        # IF NAN, DL IT
        # RETURN DF
        df = self._find_dataset(selected_symbols, selected_features, start_date, end_date, freq)
        if df is None:
            print("Dataset not found, going to download...")
            df = self._get_assets_df(selected_symbols, selected_features, start_date, end_date, freq)
            print("Download done...")
            metadata = {
                "quote_asset": self.quote_asset,
                "selected_symbols": selected_symbols,
                "selected_features": selected_features,
                "start_date": start_date,
                "end_date": end_date,
                "freq": freq,
            }

            path = Path(self.database_path)
            folder_nb = len(list(path.glob("*"))) + 1
            path = path / (self.DATASET_FOLDER_PREFIX + str(folder_nb))
            path.mkdir()
            print(f"Saving Dataframe into {path}")
            with open(path / self.METADATA_FILENAME, 'w') as outfile:
                json.dump(metadata, outfile, indent=2)
            df.to_parquet(path / self.DATA_FILENAME)
        else:
            print("Found dataset, skip download")
        return df
