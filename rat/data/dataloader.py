from typing import List

import numpy as np
import pandas as pd
from .data_manager import DataManager
from .replay_buffer import ReplayBuffer


class DataMatrices:
    def __init__(self,
                 database_path: str,
                 quote_asset: str,
                 selected_symbols: List[str],
                 selected_features: List[str],
                 date_start: str,
                 date_end: str,
                 freq: str,
                 window_size: int,
                 batch_size: int = 50,
                 buffer_bias_ratio: float = 0):
        """
        :param start: Unix time
        :param end: Unix time
        :param access_freq: the old_data access freq of the input matrix.
        :param trade_freq: the trading freq of the agent.
        :param global_freq: the old_data access freq of the global price matrix.
                              if it is not equal to the access freq, there will be inserted observations
        :param coin_filter: number of coins that would be selected
        :param window_size: freqs of input old_data
        :param train_portion: portion of training set
        :param is_permed: if False, the sample inside a mini-batch is in order
        :param validation_portion: portion of cross-validation set
        :param test_portion: portion of test set
        :param portion_reversed: if False, the order to sets are [train, validation, test]
        else the order is [test, validation, train]
        """

        # assert window_size >= MIN_NUM_PERIOD
        self.selected_symbols = selected_symbols
        self.selected_features = selected_features
        self.__history_manager = DataManager(database_path,
                                             quote_asset,
                                             selected_symbols,
                                             selected_features,
                                             date_start,
                                             date_end,
                                             freq)
        self.__global_data = self.__history_manager.get_data()
        self.indices = np.array([i for i in range(len(self.__global_data) - window_size)])
        self.__freq_length = freq
        self.__PVM = pd.DataFrame(index=self.__global_data.index,
                                  columns=self.__global_data.columns.get_level_values(DataManager.COIN_INDEX_NAME)
                                  .unique())
        self.__PVM = self.__PVM.fillna(1.0 / len(self.selected_symbols))

        self._window_size = window_size
        self._num_freqs = len(self.__global_data.index)

        self.__batch_size = batch_size
        self.__delta = 0  # the count of global increased
        self.__replay_buffer = ReplayBuffer(start_index=0,
                                            end_index=len(self.__global_data.index) - window_size,
                                            sample_bias=buffer_bias_ratio,
                                            batch_size=self.__batch_size,
                                            coin_number=len(self.selected_symbols),
                                            is_permed=False)

        print("the number of training examples is %s"
              "," % self._num_freqs)

    @property
    def global_weights(self):
        return self.__PVM

    @property
    def global_matrix(self):
        return self.__global_data

    @property
    def coin_list(self):
        return self.selected_symbols

    def get_set(self):
        return self.__pack_samples(self.indices)

    def get_test_set_online(self, ind_start, ind_end, x_window_size):
        return self.__pack_samples_test_online(ind_start, ind_end, x_window_size)

    ##############################################################################
    def next_batch(self):
        """
        @:return: the next batch of training sample. The sample is a dictionary
        with key "X"(input old_data); "y"(future relative price); "last_w" a numpy array
        with shape [batch_size, assets]; "w" a list of numpy arrays list length is
        batch_size
        """
        batch = self.__pack_samples([exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        return batch

    def __pack_samples(self, indexs):
        indexs = np.array(indexs)
        last_w = self.__PVM.values[indexs - 1, :]

        def setw(w):
            self.__PVM.iloc[indexs, :] = w

        M = [self.__get_submatrix(index) for index in indexs]
        M = np.array(M)
        X = M[:, :, :, :-1]
        y = M[:, :, :, -1] / M[:, 0, None, :, -2]  # divide by the close price (O) of timestamp before (-2)
        return {"X": X, "y": y, "last_w": last_w, "setw": setw}

    def __pack_samples_test_online(self, ind_start, ind_end, x_window_size):
        last_w = self.__PVM.values[ind_start-1:ind_start, :]

        def setw(w):
            self.__PVM.iloc[ind_start, :] = w

        M = [self.__get_submatrix_test_online(ind_start, ind_end)]  # [1,4,11,2807]
        M = np.array(M)
        X = M[:, :, :, :-1]
        y = M[:, :, :, x_window_size:] / M[:, 0, None, :, x_window_size - 1:-1]
        return {"X": X, "y": y, "last_w": last_w, "setw": setw}

    def __get_submatrix(self, ind):
        return self.__global_data[ind:ind + self._window_size + 1].values.\
            reshape(-1, len(self.selected_symbols), len(self.selected_features)).transpose(2, 1, 0)

    def __get_submatrix_test_online(self, ind_start, ind_end):
        return self.__global_data[ind_start:ind_end].values.\
            reshape(-1, len(self.selected_symbols), len(self.selected_features)).transpose(2, 1, 0)
