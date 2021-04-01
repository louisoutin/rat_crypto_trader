from rat.data.data_manager import DataManager


def test_coinlist():
    print("")
    c = DataManager("/home/luisao/perso/rat_crypto_trader/datasets",
                    ["ETHBTC", "LTCBTC"],
                    ["close", "high", "low", "open"],
                    "2021-01-01",
                    "2021-03-01",
                    "30T")
    print(c.data)
    #print(c.klines)
