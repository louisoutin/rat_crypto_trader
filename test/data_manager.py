from rat.data.data_manager import DataManager
import pandas as pd


def test_coinlist():
    print("")
    c = DataManager("/home/luisao/perso/rat_crypto_trader/datasets",
                    "USDT",
                    ["BTCUSDT", "LTCUSDT", "ETHUSDT"],
                    ["close", "high", "low", "open"],
                    "2020-02-01",
                    "2020-02-04",
                    "12h")
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(c.data)
    assert not c.data.isnull().values.any()
