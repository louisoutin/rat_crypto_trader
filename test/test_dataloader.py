from rat.data.dataloader import DataMatrices


def test_coinlist():
    d = DataMatrices(database_path="/home/luisao/perso/rat_crypto_trader/datasets",
                     quote_asset="BTC",
                     selected_symbols=["ETHBTC", "LTCBTC", "BTCUSDT"],
                     selected_features=["close", "high", "low", "open"],
                     date_start="2019-01-01",
                     date_end="2021-03-01",
                     freq="30T",
                     window_size=31,
                     batch_size=64,
                     buffer_bias_ratio=5e-5)
    print("d", d.global_matrix)
    set = d.next_batch()
    print(set)
