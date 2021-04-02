from rat.data.data_manager import DataManager
import pandas as pd


def test_coinlist():
    print("")
    c = DataManager("/home/luisao/perso/rat_crypto_trader/datasets",
                    "BTC",
                    ["ETHBTC", "LTCBTC", "BTCUSDT"],
                    ["close", "high", "low", "open"],
                    "2021-01-01",
                    "2021-02-01",
                    "30T")
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(c.data)
