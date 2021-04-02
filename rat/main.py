import os
import sys
import json
import pandas as pd
from pathlib import Path
import torch
from . import __version__
from docopt import docopt

from rat.data.dataloader import DataMatrices
from rat.helpers import make_model
from rat.loss.batch_loss import Batch_Loss
from rat.loss.test_loss import Test_Loss
from rat.loss.optimizer import NoamOpt
from rat.train import train_net
from rat.test import test_net


def run_main():
    """
    Relation-Aware Transformer for Portfolio Policy Learning

    Usage:
      rat train <config_file>
      rat test <config_file>

      rat -h | --help
      rat --version

    Options:
      -h --help     Show this screen.
      --version     Show version.
    """

    if len(sys.argv) == 1:
        sys.argv.append("--help")
    arguments = docopt(
        run_main.__doc__,
        version="RAT v.%s - Relation-Aware Transformer for Portfolio Policy Learning" % __version__,
    )

    config_file = Path(arguments["<config_file>"])
    with config_file.open("r") as fhandle:
        ctx = json.load(fhandle)

    if arguments["train"]:
        launch_train(ctx)
    if arguments["test"]:
        launch_test(ctx)


def launch_train(ctx: dict):
    if not (Path(ctx["model_dir"]) / ctx["model_name"] / str(ctx["model_index"])).exists():
        (Path(ctx["model_dir"]) / ctx["model_name"] / str(ctx["model_index"])).mkdir(parents=True)
    DM_train = DataMatrices(database_path=ctx["database_path"],
                            selected_symbols=ctx["selected_symbols"],
                            selected_features=ctx["selected_features"],
                            date_start=ctx["train_range"]["start"],
                            date_end=ctx["train_range"]["end"],
                            freq=ctx["freq"],
                            window_size=ctx["x_window_size"],
                            batch_size=ctx["batch_size"],
                            buffer_bias_ratio=ctx["buffer_bias_ratio"])

    DM_val = DataMatrices(database_path=ctx["database_path"],
                          selected_symbols=ctx["selected_symbols"],
                          selected_features=ctx["selected_features"],
                          date_start=ctx["val_range"]["start"],
                          date_end=ctx["val_range"]["end"],
                          freq=ctx["freq"],
                          window_size=ctx["x_window_size"],
                          batch_size=ctx["batch_size"],
                          buffer_bias_ratio=ctx["buffer_bias_ratio"])

    #################set learning rate###################
    lr_model_sz = 5120
    factor = ctx["learning_rate"]  # 1.0
    warmup = 0  # 800

    total_step = ctx["total_step"]
    x_window_size = ctx["x_window_size"]  # 31

    batch_size = ctx["batch_size"]
    coin_num = len(ctx["selected_symbols"])  # 11
    feature_number = len(ctx["selected_features"])  # 4
    trading_consumption = ctx["trading_consumption"]  # 0.0025
    variance_penalty = ctx["variance_penalty"]  # 0 #0.01
    cost_penalty = ctx["cost_penalty"]  # 0 #0.01
    output_step = ctx["output_step"]  # 50
    local_context_length = ctx["local_context_length"]
    model_dim = ctx["model_dim"]
    weight_decay = ctx["weight_decay"]
    interest_rate = ctx["daily_interest_rate"] / 24 / 2

    device = ctx["device"]

    model = make_model(batch_size, coin_num, x_window_size, feature_number,
                       N=1, d_model_Encoder=ctx["multihead_num"] * model_dim,
                       d_model_Decoder=ctx["multihead_num"] * model_dim,
                       d_ff_Encoder=ctx["multihead_num"] * model_dim,
                       d_ff_Decoder=ctx["multihead_num"] * model_dim,
                       h=ctx["multihead_num"],
                       dropout=0.01,
                       local_context_length=local_context_length,
                       device=device)

    # model = make_model3(N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)
    model.to(device)
    # model_size, factor, warmup, optimizer)
    model_opt = NoamOpt(lr_model_sz, factor, warmup,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9,
                                         weight_decay=weight_decay))

    loss_compute = Batch_Loss(trading_consumption, interest_rate, variance_penalty, cost_penalty, device=device)
    evaluate_loss_compute = Batch_Loss(trading_consumption, interest_rate, variance_penalty, cost_penalty,
                                       device=device)

    ##########################train net####################################################
    tst_loss, tst_portfolio_value = train_net(DM_train, DM_val, total_step, output_step, x_window_size,
                                              local_context_length, model,
                                              ctx["model_dir"], ctx["model_index"], loss_compute, evaluate_loss_compute,
                                              model_opt, device=device)

    print("best tst_loss", tst_loss)
    print("best tst_portfolio_value", tst_portfolio_value)


def launch_test(ctx):
    start = parse_time(ctx["start"])
    end = parse_time(ctx["end"])
    x_window_size = ctx["x_window_size"]  # 31
    trading_consumption = ctx["trading_consumption"]  # 0.0025
    variance_penalty = ctx["variance_penalty"]  # 0 #0.01
    cost_penalty = ctx["cost_penalty"]  # 0 #0.01
    local_context_length = ctx["local_context_length"]
    interest_rate = ctx["daily_interest_rate"] / 24 / 2

    device = ctx["device"]

    csv_dir = ctx["log_dir"] + "/" + "train_summary.csv"

    if not Path(ctx["log_dir"]).exists():
        Path(ctx["log_dir"]).mkdir(parents=True)

    model = torch.load(ctx["model_dir"] + '/' + str(ctx["model_index"]) + '.pkl')

    DM = DataMatrices(database_path=ctx["database_path"],
                      start=start, end=end,
                      market="poloniex",
                      feature_number=ctx["feature_number"],
                      window_size=ctx["x_window_size"],
                      online=False,
                      period=1800,
                      coin_filter=11,
                      is_permed=False,
                      buffer_bias_ratio=5e-5,
                      batch_size=ctx["batch_size"],  # 128,
                      volume_average_days=30,
                      test_portion=ctx["test_portion"],  # 0.08,
                      portion_reversed=False)

    test_loss_compute = Test_Loss(trading_consumption, interest_rate, variance_penalty, cost_penalty,
                                  size_average=False, device=device)

    ##########################test net#####################################################
    tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO = test_net(DM, x_window_size, local_context_length, model,
                                                                   test_loss_compute, device=device)

    d = {"net_dir": [ctx["model_index"]],
         "fAPV": [tst_portfolio_value.item()],
         "SR": [SR.item()],
         "CR": [CR.item()],
         "TO": [TO.item()],
         "St_v": [''.join(str(e) + ', ' for e in St_v)],
         "backtest_test_history": [''.join(str(e) + ', ' for e in tst_pc_array.cpu().numpy())],
         }
    new_data_frame = pd.DataFrame(data=d).set_index("net_dir")
    if os.path.isfile(csv_dir):
        dataframe = pd.read_csv(csv_dir).set_index("net_dir")
        dataframe = dataframe.append(new_data_frame)
    else:
        dataframe = new_data_frame
    dataframe.to_csv(csv_dir)


if __name__ == "__main__":
    run_main()
