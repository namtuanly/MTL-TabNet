import math

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from mmocr.models.builder import ENCODERS
from .base_encoder import BaseEncoder

@ENCODERS.register_module()
class PositionalEncoding(BaseEncoder):
    """ Implement the PE function. """

    def __init__(self, d_model, dropout=0., max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, feat, **kwargs):
        if len(feat.shape) > 3:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h*w) # flatten 2D feature map
            feat = feat.permute((0,2,1))
        feat = feat + self.pe[:, :feat.size(1)] # pe 1*5000*512
        return self.dropout(feat)

    def init_weights(self):
        pass


# namly
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


@ENCODERS.register_module()
class PositionalEncoding2D(BaseEncoder):
    def __init__(self, d_model, dropout=0., max_len=5000):
        """
        namly
        :param d_model: The last dimension of the tensor you want to apply pos emb to.
        : refer: https://github.com/tatp22/multidim-positional-encoding
        """
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.org_channels = d_model
        channels = int(np.ceil(d_model / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, feat, **kwargs):
        """
        :param feat: A 4d tensor of size (batch_size, ch, x, y)
        :return: Positional Encoding Matrix of size (batch_size, ch, x, y)
        """

        if len(feat.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        # feat: A 4d tensor of size (batch_size, ch, x, y)
        batch_size, orig_ch, x, y = feat.shape

        if self.cached_penc is None or self.cached_penc.shape != feat.shape:
            # change share of input: (batch_size, ch, x, y) -> (batch_size, x, y, ch)
            feat = feat.permute(0, 2, 3, 1)

            self.cached_penc = None

            pos_x = torch.arange(x, device=feat.device).type(self.inv_freq.type())
            pos_y = torch.arange(y, device=feat.device).type(self.inv_freq.type())
            sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
            sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
            emb_x = get_emb(sin_inp_x).unsqueeze(1)
            emb_y = get_emb(sin_inp_y)
            emb = torch.zeros((x, y, self.channels * 2), device=feat.device).type(
                feat.type()
            )
            emb[:, :, : self.channels] = emb_x
            emb[:, :, self.channels : 2 * self.channels] = emb_y

            self.cached_penc = emb[None, :, :, :orig_ch].repeat(feat.shape[0], 1, 1, 1)

            # change share of output: (batch_size, x, y, ch) -> (batch_size, ch, x, y)
            self.cached_penc = self.cached_penc.permute(0, 3, 1, 2)
            feat = feat.permute(0, 3, 1, 2)

        # feat: A 4d tensor of size (batch_size, ch, x, y)
        feat = feat + self.cached_penc
        feat = feat.view(batch_size, orig_ch, x * y)  # flatten 2D feature map
        feat = feat.permute((0, 2, 1))
        return self.dropout(feat)

    def init_weights(self):
        pass
