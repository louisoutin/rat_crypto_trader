import pandas as pd
from rat.data.history_manager import HistoryManager
# from rat.data.constants import *


def test_history():
    start_date = pd.to_datetime(['2017-02-01'])
    start_date = ((start_date - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))[0]
    end_date = pd.to_datetime(['2017-04-10'])
    end_date = ((end_date - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))[0]
    h = HistoryManager(coin_number=5,
                       end=end_date,
                       volume_average_days=30,
                       volume_forward=0,
                       online=False
                       )
    dataframe = h.get_global_panel(start_date, end_date, 1800, ["close", "high", "low", "open"])
    print(dataframe)
