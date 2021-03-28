from rat.old_data.exchange.poloniex import Poloniex


def test_poloniex():
    p = Poloniex()
    print(p.marketVolume())