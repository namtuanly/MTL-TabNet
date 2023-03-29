import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.models.builder import DECODERS
from .base_decoder import BaseDecoder
from ..encoders.positional_encoding import PositionalEncoding

from mmocr.models.builder import DECODERS
from .local_attention import LocalAttention

class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input):
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)


def clones(module, N):
    """ Produce N identical layers """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SubLayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        #tmp = self.norm(x)
        #tmp = sublayer(tmp)
        return x + self.dropout(sublayer(self.norm(x)))


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def self_attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scale Dot Product Attention'
    query, key: (bsz, num_heads, seq_len, head_dim)
    """

    d_k = value.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))
    if mask is not None:
        #score = score.masked_fill(mask == 0, -1e9) # b, h, L, L
        score = score.masked_fill(mask == 0, -6.55e4) # for fp16
    p_attn = F.softmax(score, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadLocalAttention(nn.Module):
    # Local Self-attention in LongFormer

    def __init__(self, headers, d_model, dropout, structure_window=0, cell_window=0):
        super(MultiHeadLocalAttention, self).__init__()

        assert d_model % headers == 0
        self.d_k = int(d_model / headers)
        self.headers = headers
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.local_attention = LocalAttention(dim=self.d_k, window_size=structure_window, causal=True,
                                              look_backward=1, look_forward=0, dropout=dropout,
                                              autopad=True, exact_windowsize=False)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.headers, self.d_k).transpose(1, 2)
             for l,x in zip(self.linears, (query, key, value))]
        # 2) Apply LocalAttention on all the projected vectors in batch

        x = self.local_attention.forward(query, key, value, mask=mask.bool())
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.headers * self.d_k)
        return self.linears[-1](x)


class MultiHeadGlobalAttention(nn.Module):

    def __init__(self, headers, d_model, dropout, window_size=0):
        super(MultiHeadGlobalAttention, self).__init__()

        assert d_model % headers == 0
        self.d_k = int(d_model / headers)
        self.headers = headers
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.headers, self.d_k).transpose(1, 2)
             for l,x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = self_attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.headers * self.d_k)
        return self.linears[-1](x)


class DecoderLayer(nn.Module):
    """
    Decoder is made of self attention, srouce attention and feed forward.
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = MultiHeadLocalAttention(**self_attn)
        self.src_attn = MultiHeadGlobalAttention(**src_attn)
        self.feed_forward = FeedForward(**feed_forward)
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)

    def forward(self, x, feature, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, feature, feature, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadLocalAttentionCell(nn.Module):
    # Local Self-attention in LongFormer

    def __init__(self, headers, d_model, dropout, structure_window=0, cell_window=0):
        super(MultiHeadLocalAttentionCell, self).__init__()

        assert d_model % headers == 0
        self.d_k = int(d_model / headers)
        self.headers = headers
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.local_attention = LocalAttention(dim=self.d_k, window_size=cell_window, causal=True,
                                              look_backward=1, look_forward=0, dropout=dropout,
                                              autopad=True, exact_windowsize=False)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.headers, self.d_k).transpose(1, 2)
             for l,x in zip(self.linears, (query, key, value))]
        # 2) Apply LocalAttention on all the projected vectors in batch

        x = self.local_attention.forward(query, key, value, mask=mask.bool())
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.headers * self.d_k)
        return self.linears[-1](x)


class MultiHeadGlobalAttentionCell(nn.Module):
    # namly
    # MultiHeadCrossAttention in CellContentDecoder
    # reduce the memory

    def __init__(self, headers, d_model, dropout):
        super(MultiHeadGlobalAttentionCell, self).__init__()

        assert d_model % headers == 0
        self.d_k = int(d_model / headers)
        self.headers = headers
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x) for l,x in zip(self.linears, (query, key, value))]

        query = query.view(nbatches, -1, self.headers, self.d_k).transpose(1, 2)
        # print(key.size())
        key = key.contiguous().view(1, -1, self.headers, self.d_k).transpose(1, 2)
        value = value.contiguous().view(1, -1, self.headers, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = self_attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.headers * self.d_k)
        return self.linears[-1](x)


class DecoderLayerCell(nn.Module):
    """
    For CellContentDecoder
    Decoder is made of self attention, srouce attention and feed forward.
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayerCell, self).__init__()
        self.size = size
        self.self_attn = MultiHeadLocalAttentionCell(**self_attn)
        self.src_attn = MultiHeadGlobalAttentionCell(**src_attn)
        self.feed_forward = FeedForward(**feed_forward)
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)

    def forward(self, x, feature, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, feature, feature, src_mask))
        return self.sublayer[2](x, self.feed_forward)


@DECODERS.register_module()
class MasterDecoder(BaseDecoder):

    def __init__(self,
                 N,
                 decoder,
                 d_model,
                 num_classes,
                 start_idx,
                 padding_idx,
                 max_seq_len,
                 ):
        super(MasterDecoder, self).__init__()
        self.layers = clones(DecoderLayer(**decoder), N)
        self.norm = nn.LayerNorm(decoder.size)
        self.fc = nn.Linear(d_model, num_classes)

        self.embedding = Embeddings(d_model=d_model, vocab=num_classes)
        self.positional_encoding = PositionalEncoding(d_model=d_model)

        self.SOS = start_idx
        self.PAD = padding_idx
        self.max_length = max_seq_len

    def make_mask(self, src, tgt):
        """
        Make mask for self attention.
        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).byte()

        tgt_len = tgt.size(1)
        trg_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8, device=src.device))

        tgt_mask = trg_pad_mask & trg_sub_mask
        return None, tgt_mask

    def decode(self, input, feature, src_mask, tgt_mask):
        # main process of transformer decoder.
        x = self.embedding(input)
        x = self.positional_encoding(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, feature, src_mask, tgt_mask)
        x = self.norm(x)
        return self.fc(x)

    def greedy_forward(self, SOS, feature, mask):
        input = SOS
        output = None
        for i in range(self.max_length+1):
            _, target_mask = self.make_mask(feature, input)
            out = self.decode(input, feature, None, target_mask)
            #out = self.decoder(input, feature, None, target_mask)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        return output

    def forward_train(self, feat, out_enc, targets_dict, img_metas=None):
        # x is token of label
        # feat is feature after backbone before pe.
        # out_enc is feature after pe.
        device = feat.device
        if isinstance(targets_dict, dict):
            padded_targets = targets_dict['padded_targets'].to(device)
        else:
            padded_targets = targets_dict.to(device)

        src_mask = None
        _, tgt_mask = self.make_mask(out_enc, padded_targets[:,:-1])
        return self.decode(padded_targets[:, :-1], out_enc, src_mask, tgt_mask)

    def forward_test(self, feat, out_enc, img_metas):
        src_mask = None
        batch_size = out_enc.shape[0]
        SOS = torch.zeros(batch_size).long().to(out_enc.device)
        SOS[:] = self.SOS
        SOS = SOS.unsqueeze(1)
        output = self.greedy_forward(SOS, out_enc, src_mask)
        return output

    def forward(self,
                feat,
                out_enc,
                targets_dict=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets_dict, img_metas)

        return self.forward_test(feat, out_enc, img_metas)


@DECODERS.register_module()
class TableMasterDecoder(BaseDecoder):
    """
    Split to two transformer header at the last layer.
    Cls_layer is used to structure token classification.
    Bbox_layer is used to regress bbox coord.
    """
    def __init__(self,
                 N,
                 decoder,
                 d_model,
                 num_classes,
                 num_classes_cell,  # number classes in cell content
                 start_idx,
                 padding_idx,
                 max_seq_len,
                 start_idx_cell,
                 padding_idx_cell,
                 max_seq_len_cell,
                 idx_tag_cell
                 ):
        super(TableMasterDecoder, self).__init__()
        self.layers = clones(DecoderLayer(**decoder), N-1)
        self.cls_layer = clones(DecoderLayer(**decoder), 1)
        self.bbox_layer = clones(DecoderLayer(**decoder), 1)
        self.cell_layer = clones(DecoderLayerCell(**decoder), 1)          # cell content classification
        self.cls_fc = nn.Linear(d_model, num_classes)
        self.bbox_fc = nn.Sequential(
            nn.Linear(d_model, 4),
            nn.Sigmoid()
        )
        self.cell_fc = nn.Linear(d_model, num_classes_cell)             # cell content classification
        self.norm = nn.LayerNorm(decoder.size)
        self.embedding = Embeddings(d_model=d_model, vocab=num_classes)
        self.positional_encoding = PositionalEncoding(d_model=d_model)

        # namly
        # embedding and positional encoding for cell content decoder
        self.embedding_cell = Embeddings(d_model=d_model, vocab=num_classes_cell)
        self.positional_encoding_cell = PositionalEncoding(d_model=d_model)

        self.SOS_CELL = start_idx_cell
        self.PAD_CELL = padding_idx_cell
        self.max_length_cell = max_seq_len_cell
        self.cell_input_fc = nn.Linear(2 * d_model, d_model)
        self.batch_size_cell = 32
        # vaskar
        # batch_size = 4, batch_size_cell = 32       cell 150 batch 4
        # batch_size = 4, batch_size_cell = 64       cell 100 batch 4

        # heliar
        # batch_size = 4, batch_size_cell = 32       cell 100 batch 4 train and val

        # tag id of non empty cells
        self.idx_tag_cell = idx_tag_cell
        # print_nam
        # print(self.SOS_CELL)
        # print(self.PAD_CELL)
        # print(self.max_length_cell)
        # print(self.idx_tag_cell)
        # print(num_classes_cell)

        self.SOS = start_idx
        self.PAD = padding_idx
        self.max_length = max_seq_len

    def make_mask_structure(self, src, tgt):
        """
        Make mask for self attention in the structure decoder.
        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        trg_pad_mask = (tgt != self.PAD).byte()

        return None, trg_pad_mask

    def make_mask_cell(self, src, tgt):
        """
        Make mask for self attention in the cell content decoder.
        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        trg_pad_mask = (tgt != self.PAD_CELL).byte()

        return None, trg_pad_mask

    def decode(self, input, feature, src_mask, input_cell_content, bbox_masks):
        # input: batch_size * self.max_seq_len
        # feature: batch_size * (w * h) * d_feature
        # main process of transformer decoder.
        x = self.embedding(input)
        x = self.positional_encoding(x)
        _, target_mask = self.make_mask_structure(feature, input)

        # origin transformer layers
        # x: tensor(batch_size, max_seq_len, d_model)
        for i, layer in enumerate(self.layers):
            x = layer(x, feature, src_mask, target_mask)

        # cls head
        # cls_x: batch_size * max_seq_len * d_model
        for layer in self.cls_layer:
            cls_x = layer(x, feature, src_mask, target_mask)
        cls_x = self.norm(cls_x)

        # bbox head
        # bbox_x: batch_size * max_seq_len * d_model
        for layer in self.bbox_layer:
            bbox_x = layer(x, feature, src_mask, target_mask)
        bbox_x = self.norm(bbox_x)

        # namly
        # cell content head
        # main process of transformer decoder.
        # input_cell_content: (list[list[Tensor]]) batch_size, sample_seq_len, self.max_seq_len_cell
        device = feature.device

        cell_x_out = []
        for idx_, input_sample_padded_target in enumerate(input_cell_content):
            input_sample_padded_target = torch.stack(input_sample_padded_target, 0).long().to(device)[:, :-1]

            x_cell = self.embedding_cell(input_sample_padded_target)
            x_cell = self.positional_encoding_cell(x_cell)
            _, cell_target_mask = self.make_mask_cell(feature, input_sample_padded_target)

            # feature
            feature_i = feature[idx_].unsqueeze(0)
            # feature_i = feature_i.expand(1, -1, -1)

            # x from origin transformer layers
            # x: tensor(batch_size, max_seq_len, d_model)
            bbox_masks_i = bbox_masks[idx_].to(device)[1:].to(torch.bool)
            x_i = x[idx_, bbox_masks_i].unsqueeze(1)
            x_i = x_i.expand(-1, self.max_length_cell - 1, -1)
            x_cell_i = self.cell_input_fc(torch.cat((x_cell, x_i), -1))

            cell_x = None
            for i_batch in range(math.ceil(input_sample_padded_target.size(0) / self.batch_size_cell)):
                # calculate start_i and end_i
                start_i = i_batch * self.batch_size_cell
                end_i = (i_batch + 1) * self.batch_size_cell
                if (i_batch + 1) == math.ceil(input_sample_padded_target.size(0) / self.batch_size_cell):
                    end_i = input_sample_padded_target.size(0)

                for layer in self.cell_layer:
                    cell_x_batch = layer(x_cell_i[start_i:end_i], feature_i, None,
                                         cell_target_mask[start_i:end_i])
                cell_x_batch = self.norm(cell_x_batch)

                # concat each batch_cell one tensor
                if cell_x is None:
                    cell_x = cell_x_batch
                else:
                    cell_x = torch.cat((cell_x, cell_x_batch), 0)

            # FC and append to batch_size
            cell_x_out.append(self.cell_fc(cell_x))

        return self.cls_fc(cls_x), self.bbox_fc(bbox_x), cell_x_out

    def decode_test(self, input, feature, src_mask, decode_cell=False):
        # author: namly
        # main process of transformer decoder.
        x = self.embedding(input)
        x = self.positional_encoding(x)
        _, target_mask = self.make_mask_structure(feature, input)

        # origin transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, feature, src_mask, target_mask)

        # cls head
        for layer in self.cls_layer:
            cls_x = layer(x, feature, src_mask, target_mask)
        cls_x = self.norm(cls_x)
        tag_x_out = self.cls_fc(cls_x)

        # bbox head
        for layer in self.bbox_layer:
            bbox_x = layer(x, feature, src_mask, target_mask)
        bbox_x = self.norm(bbox_x)

        # namly
        # cell content head
        # main process of transformer decoder.
        # input_cell_content: (list[list[Tensor]]) batch_size, sample_seq_len, self.max_seq_len_cell
        # bbox_masks: tensor: batch_size * max_seq_len
        cell_x_out = []
        if decode_cell:
            # calculate book_masks
            # TODO check book_masks
            prob_tag = F.softmax(tag_x_out, dim=-1)
            _, out_tag_decode = torch.max(prob_tag, dim=-1)
            bbox_masks = torch.where((out_tag_decode == self.idx_tag_cell[0]) |
                                     (out_tag_decode == self.idx_tag_cell[1]), 1, 0)

            device = feature.device
            # decoder cell content for each sample in batch_size
            for idx_, bbox_masks_i in enumerate(bbox_masks):
                if torch.count_nonzero(bbox_masks_i).item() == 0:
                    output_cell = torch.zeros(1)
                    cell_x_out.append(output_cell)
                    continue

                # x from origin transformer layers
                # x: tensor(batch_size, max_seq_len, d_model)
                bbox_masks_i = bbox_masks_i.to(torch.bool)
                x_i = x[idx_, bbox_masks_i].unsqueeze(1)     # number_cell * 1 * d_model

                # init the input of cell decoder
                SOS = torch.zeros(x_i.shape[0]).long().to(device)
                SOS[:] = self.SOS_CELL
                SOS = SOS.unsqueeze(1)
                input_cell = SOS     # number_cell * 1

                # decoder each step of cell content
                for i_step in range(self.max_length_cell + 1):
                    x_cell = self.embedding_cell(input_cell)
                    x_cell = self.positional_encoding_cell(x_cell)
                    _, cell_target_mask = self.make_mask_cell(feature, input_cell)

                    # feature
                    feature_i = feature[idx_].unsqueeze(0)
                    # feature_i = feature_i.expand(1, -1, -1)

                    # concat x_i from origin transformer layers and x_cell from input of cell_decoder
                    x_i_step = x_i.expand(-1, i_step + 1, -1)  # number_cell * (i_step + 1) * d_model
                    x_cell_i = self.cell_input_fc(torch.cat((x_cell, x_i_step), -1))

                    cell_x = None
                    for i_batch in range(math.ceil(input_cell.size(0) / self.batch_size_cell)):
                        # calculate start_i and end_i
                        start_i = i_batch * self.batch_size_cell
                        end_i = (i_batch + 1) * self.batch_size_cell
                        if (i_batch + 1) == math.ceil(input_cell.size(0) / self.batch_size_cell):
                            end_i = input_cell.size(0)

                        for layer in self.cell_layer:
                            cell_x_batch = layer(x_cell_i[start_i:end_i], feature_i, None,
                                                 cell_target_mask[start_i:end_i])
                        cell_x_batch = self.norm(cell_x_batch)

                        # concat each batch_cell one tensor
                        if cell_x is None:
                            cell_x = cell_x_batch
                        else:
                            cell_x = torch.cat((cell_x, cell_x_batch), 0)

                    out_cell = self.cell_fc(cell_x)
                    output_cell = out_cell
                    prob_cell = F.softmax(out_cell, dim=-1)
                    _, next_word = torch.max(prob_cell, dim=-1)
                    input_cell = torch.cat([input_cell, next_word[:, -1].unsqueeze(-1)], dim=1)

                # FC and append to batch_size
                cell_x_out.append(output_cell)

        return tag_x_out, self.bbox_fc(bbox_x), cell_x_out

    def greedy_forward(self, SOS, feature, mask):
        input = SOS
        output = None
        # author: namly
        for i in range(self.max_length+1):
            if i == self.max_length:
                out, bbox_output, cell_output = self.decode_test(input, feature, None, True)
                output = out
                break
            else:
                out, bbox_output, cell_output = self.decode_test(input, feature, None, False)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        return output, bbox_output, cell_output

    def forward_train(self, feat, out_enc, targets_dict, img_metas=None):
        # x is token of label
        # feat is feature after backbone before pe.
        # out_enc is feature after pe.
        device = feat.device

        assert isinstance(targets_dict, dict)
        padded_targets = targets_dict['padded_targets'].to(device)  # padded_targets: batch_size * self.max_seq_len
        cell_padded_targets = targets_dict['cell_padded_targets']   #.to(device)
        # cell_padded_targets: (list[list[Tensor]]) batch_size, sample_seq_len, self.max_seq_len_cell

        bbox_masks = targets_dict['bbox_masks']
        # bbox_masks: tensor(batch_size, max_seq_len)

        src_mask = None
        return self.decode(padded_targets[:, :-1], out_enc, src_mask, cell_padded_targets, bbox_masks)

    def forward_test(self, feat, out_enc, img_metas):
        src_mask = None
        batch_size = out_enc.shape[0]
        SOS = torch.zeros(batch_size).long().to(out_enc.device)
        SOS[:] = self.SOS
        SOS = SOS.unsqueeze(1)
        output, bbox_output, cell_output = self.greedy_forward(SOS, out_enc, src_mask)
        return output, bbox_output, cell_output

    def forward(self,
                feat,
                out_enc,
                targets_dict=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets_dict, img_metas)

        return self.forward_test(feat, out_enc, img_metas)


@DECODERS.register_module()
class TableMasterConcatDecoder(BaseDecoder):
    """
    Split to two transformer header at the last layer.
    Cls_layer is used to structure token classification.
    Bbox_layer is used to regress bbox coord.
    """
    def __init__(self,
                 N,
                 decoder,
                 d_model,
                 num_classes,
                 start_idx,
                 padding_idx,
                 max_seq_len,
                 ):
        super(TableMasterConcatDecoder, self).__init__()
        self.layers = clones(DecoderLayer(**decoder), N-1)
        self.cls_layer = clones(DecoderLayer(**decoder), 1)
        self.bbox_layer = clones(DecoderLayer(**decoder), 1)
        self.cls_fc = nn.Linear(d_model, num_classes)
        self.bbox_fc = nn.Sequential(
            nn.Linear(d_model, 4),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(decoder.size)
        self.embedding = Embeddings(d_model=d_model, vocab=num_classes)
        self.positional_encoding = PositionalEncoding(d_model=d_model)

        self.SOS = start_idx
        self.PAD = padding_idx
        self.max_length = max_seq_len

    def make_mask(self, src, tgt):
        """
        Make mask for self attention.
        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).byte()

        tgt_len = tgt.size(1)
        trg_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8, device=src.device))

        tgt_mask = trg_pad_mask & trg_sub_mask
        return None, tgt_mask

    def decode(self, input, feature, src_mask, tgt_mask):
        # main process of transformer decoder.
        x = self.embedding(input)
        x = self.positional_encoding(x)

        # x_list = []
        cls_x_list = []
        bbox_x_list = []

        # origin transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, feature, src_mask, tgt_mask)

        # cls head
        for layer in self.cls_layer:
            cls_x = layer(x, feature, src_mask, tgt_mask)
            cls_x_list.append(cls_x)
        cls_x = torch.cat(cls_x_list, dim=-1)
        cls_x = self.norm(cls_x)

        # bbox head
        for layer in self.bbox_layer:
            bbox_x = layer(x, feature, src_mask, tgt_mask)
            bbox_x_list.append(bbox_x)
        bbox_x = torch.cat(bbox_x_list, dim=-1)
        bbox_x = self.norm(bbox_x)

        return self.cls_fc(cls_x), self.bbox_fc(bbox_x)

    def greedy_forward(self, SOS, feature, mask):
        input = SOS
        output = None
        for i in range(self.max_length+1):
            _, target_mask = self.make_mask(feature, input)
            out, bbox_output = self.decode(input, feature, None, target_mask)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        return output, bbox_output

    def forward_train(self, feat, out_enc, targets_dict, img_metas=None):
        # x is token of label
        # feat is feature after backbone before pe.
        # out_enc is feature after pe.
        device = feat.device
        if isinstance(targets_dict, dict):
            padded_targets = targets_dict['padded_targets'].to(device)
        else:
            padded_targets = targets_dict.to(device)

        src_mask = None
        _, tgt_mask = self.make_mask(out_enc, padded_targets[:,:-1])
        return self.decode(padded_targets[:, :-1], out_enc, src_mask, tgt_mask)

    def forward_test(self, feat, out_enc, img_metas):
        src_mask = None
        batch_size = out_enc.shape[0]
        SOS = torch.zeros(batch_size).long().to(out_enc.device)
        SOS[:] = self.SOS
        SOS = SOS.unsqueeze(1)
        output, bbox_output = self.greedy_forward(SOS, out_enc, src_mask)
        return output, bbox_output

    def forward(self,
                feat,
                out_enc,
                targets_dict=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets_dict, img_metas)

        return self.forward_test(feat, out_enc, img_metas)