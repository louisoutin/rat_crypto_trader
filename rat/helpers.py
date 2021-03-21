import copy
import logging

import torch
from torch import nn
import numpy as np

from rat.network.attention import MultiHeadedAttention
from rat.network.encoder_decoder import Encoder, Decoder, EncoderDecoder
from rat.network.layer.encoder import EncoderLayer
from rat.network.layer.decoder import DecoderLayer
from rat.network.positional_encoding import PositionwiseFeedForward, PositionalEncoding


def make_model(batch_size, coin_num, window_size, feature_number, N=6,
               d_model_Encoder=512, d_model_Decoder=16, d_ff_Encoder=2048, d_ff_Decoder=64, h=8, dropout=0.0,
               local_context_length=3, device="cpu"):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn_Encoder = MultiHeadedAttention(True, h, d_model_Encoder, 0.1, local_context_length, device)
    attn_Decoder = MultiHeadedAttention(True, h, d_model_Decoder, 0.1, local_context_length, device)
    attn_En_Decoder = MultiHeadedAttention(False, h, d_model_Decoder, 0.1, 1, device)
    ff_Encoder = PositionwiseFeedForward(d_model_Encoder, d_ff_Encoder, dropout)
    ff_Encoder.to(device)
    ff_Decoder = PositionwiseFeedForward(d_model_Decoder, d_ff_Decoder, dropout)
    ff_Decoder.to(device)
    position_Encoder = PositionalEncoding(d_model_Encoder, 0, dropout)
    position_Encoder.to(device)
    position_Decoder = PositionalEncoding(d_model_Decoder, window_size - local_context_length * 2 + 1, dropout)

    model = EncoderDecoder(batch_size, coin_num, window_size, feature_number, d_model_Encoder, d_model_Decoder,
                           Encoder(EncoderLayer(d_model_Encoder, c(attn_Encoder), c(ff_Encoder), dropout), N),
                           Decoder(DecoderLayer(d_model_Decoder, c(attn_Decoder), c(attn_En_Decoder), c(ff_Decoder),
                                                dropout), N),
                           c(position_Encoder),  # price series position ecoding
                           c(position_Decoder),  # local_price_context position ecoding
                           local_context_length,
                           device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    print("Parameters (param name -> param count):")
    for pname, pparams in model.named_parameters():
        pcount = np.prod(pparams.size())
        print(f"\t{pname} -> {pcount}")

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    param_count = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Total param count: {param_count}")

    return model


def train_one_step(DM, x_window_size, model, loss_compute, local_context_length, device):
    batch = DM.next_batch()
    batch_input = batch["X"]  # (128, 4, 11, 31)
    batch_y = batch["y"]  # (128, 4, 11)
    batch_last_w = batch["last_w"]  # (128, 11)
    batch_w = batch["setw"]
    #############################################################################
    previous_w = torch.tensor(batch_last_w, dtype=torch.float, device=device)
    previous_w = torch.unsqueeze(previous_w, 1)  # [128, 11] -> [128,1,11]
    batch_input = batch_input.transpose((1, 0, 2, 3))
    batch_input = batch_input.transpose((0, 1, 3, 2))
    src = torch.tensor(batch_input, dtype=torch.float, device=device)
    price_series_mask = (torch.ones(src.size()[1], 1, x_window_size) == 1)  # [128, 1, 31]
    currt_price = src.permute((3, 1, 2, 0))  # [4,128,31,11]->[11,128,31,4]
    if local_context_length > 1:
        padding_price = currt_price[:, :, -local_context_length * 2 + 1:-1, :]
    else:
        padding_price = None
    currt_price = currt_price[:, :, -1:, :]  # [11,128,31,4]->[11,128,1,4]
    trg_mask = make_std_mask(currt_price, src.size()[1])
    batch_y = batch_y.transpose((0, 2, 1))  # [128, 4, 11] ->#[128,11,4]
    trg_y = torch.tensor(batch_y, dtype=torch.float, device=device)
    out = model.forward(src, currt_price, previous_w,
                        price_series_mask, trg_mask, padding_price)
    new_w = out[:, :, 1:]  # 去掉cash
    new_w = new_w[:, 0, :]  # #[109,1,11]->#[109,11]
    new_w = new_w.detach().cpu().numpy()
    batch_w(new_w)

    loss, portfolio_value = loss_compute(out, trg_y)
    return loss, portfolio_value


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(local_price_context, batch_size):
    "Create a mask to hide padding and future words."
    local_price_mask = (torch.ones(batch_size, 1, 1) == 1)
    local_price_mask = local_price_mask & (subsequent_mask(local_price_context.size(-2)).type_as(local_price_mask.data))
    return local_price_mask
