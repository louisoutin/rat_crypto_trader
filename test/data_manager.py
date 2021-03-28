from rat.data.data_manager import DataManager


def test_coinlist():
    c = DataManager(["ETHBTC", "BTC"], "", "", "30min")
    #print(c.klines)
