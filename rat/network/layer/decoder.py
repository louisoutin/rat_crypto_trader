from torch import nn
from rat.network import clones
from .sublayer_connection import SublayerConnection


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, price_series_mask, local_price_mask, padding_price):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, local_price_mask, padding_price, padding_price))
        x = x[:, :, -1:, :]
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, price_series_mask, None, None))
        return self.sublayer[2](x, self.feed_forward)
