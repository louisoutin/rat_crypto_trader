from rat.data.dataloader import DataMatrices


def test_coinlist():
    d = DataMatrices(database_path="/home/luisao/perso/rat_crypto_trader/datasets",
                     selected_symbols=["ETHBTC", "LTCBTC"],
                     selected_features=["close", "high", "low", "open"],
                     date_start="2021-01-01",
                     date_end="2021-03-01",
                     freq="30T",
                     window_size=30,
                     batch_size=64,
                     buffer_bias_ratio=5e-5)
    print("d", d.global_matrix)
    set = d.get_set()
    print(set["X"].shape)
