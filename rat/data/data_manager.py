from typing import List
from binance.client import Client


class DataManager():

    def __init__(self,
                 selected_symbols: List[str],
                 date_start: str = "2016-01-01",
                 date_end: str = "2018-01-01",
                 period: str = "30min"):
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
        self.client = Client()
        self.klines = self.client.get_historical_klines("ETHBTC", Client.KLINE_INTERVAL_30MINUTE, "1 Dec, 2017", "1 Jan, 2018")
        info = self.client.get_exchange_info()
        symbols = [s["symbol"] for s in info["symbols"] if "BTC" in s["symbol"]]
        if not set(selected_symbols).issubset(set(symbols)):
            raise RuntimeError(f"Some selected symbols are not in the available symbols \nSelected symbols : "
                               f"\n{selected_symbols}. \nAvailables symbols: \n{symbols}")
        self.symbols = selected_symbols
        self._get_kline_df("ETHBTC")

    def _get_kline_df(self, symbol):
        klines = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_30MINUTE, "2016-01-01", "2016-02-01")
        print("klines", klines)
