# rat_crypto_trader
Relation-Aware Transformer for Portfolio Policy Learning

Repo inspired from https://github.com/Ivsxk/RAT / https://www.ijcai.org/proceedings/2020/641 with refactored code
and binance data provider.


### Install notes
Python 3.8

`pip install .`

`rat train default_configs`
`rat test default_configs`


exemple json input:

```
{
  "database_path": "path/datasets",
  "train_range": {
    "start": "2021-01-01",
    "end": "2021-03-10"
  },
  "val_range": {
    "start": "2021-03-10",
    "end": "2021-03-15"
  },
  "test_range": {
    "start": "2021-03-15",
    "end": "2021-04-01"
  },
  "quote_asset": "USDT",
  "selected_symbols": ["BTCUSDT", "LTCUSDT", "ADAUSDT", "XRPUSDT", "ZECUSDT", "XLMUSDT"],
  "selected_features": ["close", "high", "low", "open"],
  "freq": "30T",
  "total_step": 50000,
  "output_step": 1000,
  "x_window_size": 31,
  "batch_size": 256,
  "trading_consumption": 0.0025,
  "variance_penalty": 0.0,
  "cost_penalty": 0.0,
  "learning_rate": 0.0001,
  "model_dir": "path/runs",
  "model_name": "first",
  "model_index": 1,
  "model_dim": 12,
  "multihead_num": 2,
  "local_context_length": 5,
  "weight_decay": 5e-8,
  "daily_interest_rate": 0.001,
  "buffer_bias_ratio": 5e-5,
  "log_dir": "/path.logs",
  "device": "cuda:0"
}
```
