from rat.data.dataloader import DataMatrices, parse_time


def test_coinlist():
    d = DataMatrices(start=parse_time("2016-12-01"),
                     end=parse_time("2016-12-31"),
                     market="poloniex",
                     feature_number=4,
                     window_size=31,
                     online=False,
                     period=1800,
                     coin_filter=11,
                     is_permed=False,
                     buffer_bias_ratio=5e-5,
                     batch_size=128,
                     volume_average_days=30,
                     test_portion=0.08,
                     portion_reversed=False)
    print("d", d)
    d.get_submatrix(100)
