import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Transformer(nn.Module):
    def __init__(self, d_model, n_layers, dropout, n_head, n_hid, max_len):
        super(Transformer, self).__init__()
        self.d_model = d_model
        encoder_layers = TransformerEncoderLayer(d_model, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.pos_encoder = PositionalEncoding(d_model, max_len)  # arg 1 is max_doc_words
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = torch.nn.LayerNorm(d_model)
        self.max_len = max_len

    def forward(self, embedded, stc_lens, POS):
        """
        :param embedded: max_stc_len * batch * emsize / 200, 128, 300
        :param offsets: list(len=1) of 1D tensor
        :param stc_lens: [[3,2,6...]] len=1
        :return: blk_num * 1 * hidden
        """
        embedded = embedded * math.sqrt(self.d_model)
        max_len = embedded.shape[0]

        pos_embedded = embedded + torch.index_select(self.pos_encoder(torch.tensor(stc_lens).to(device), POS),
                                                     0, torch.tensor(list(range(max_len))).to(device))
        pos_embedded = self.LayerNorm(pos_embedded)
        pos_embedded = self.dropout(pos_embedded)
        # N * S: used to mark padded tokens
        key_padding_mask = self.padding_mask(stc_lens, embedded.shape[0])
        enc_sents = self.transformer_encoder(pos_embedded, src_key_padding_mask=key_padding_mask)
        return enc_sents

    @staticmethod
    def padding_mask(s_len, max_stc_len):
        # masks
        pad_mask_lst = []
        for i, stc_len in enumerate(s_len):
            mask_lst = [0] * stc_len + [1] * (max_stc_len - stc_len)
            pad_mask_lst.append(torch.tensor(mask_lst).eq(1).unsqueeze(0))
        return torch.cat(pad_mask_lst).to(device)


# --------------------------------  util models ----------------------------------------------------
# -- for attn encoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        # form PE metrics as parameters
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        # -- A::N means starts at idx A, choose every N (e.g., for [0, 1, 2, 3, 4], [0::2]->[0, 2, 4])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        # -- PE for <pad> to solve diff length
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, torch.from_numpy(position_encoding).float()))
        # form embedding layer with parameters above
        self.max_len = max_seq_len
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)
        self.seg_embeddings = nn.Embedding(2, d_model)

    def forward(self, input_len, offsets):
        """
          input_len: [BATCH_SIZE, 1]  with dim 1 as seq_len
        """
        seg_tsr = torch.tensor([[0 if x < pos else 1 for x in range(200)] for pos in offsets]).permute(1, 0).to(device)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # get input with padding
        input_pos = tensor([list(range(1, l.item() + 1))
                            + [0] * (self.max_len - l.item()) for l in input_len]).permute(1, 0)
        return self.position_encoding(input_pos) + self.seg_embeddings(seg_tsr)


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(RelativePositionalEncoding, self).__init__()
        # form PE metrics as parameters
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        # -- A::N means starts at idx A, choose every N (e.g., for [0, 1, 2, 3, 4], [0::2]->[0, 2, 4])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        # -- PE for <pad> to solve diff length
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, torch.from_numpy(position_encoding).float()))
        # form embedding layer with parameters above
        self.max_len = max_seq_len
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)
        self.seg_embeddings = nn.Embedding(2, d_model)

    def forward(self, input_len, offsets):
        """
          input_len: [BATCH_SIZE, 1]  with dim 1 as seq_len
        """
        seg_tsr = torch.tensor([[0 if x < pos else 1 for x in range(200)] for pos in offsets]).permute(1, 0).to(device)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # get input with padding
        input_pos = tensor([[(pos - x) if x < pos else (x + 1 - pos) for x in range(input_len[id_].item())]
         + [0]*(200-input_len[id_].item()) for id_, pos in enumerate(offsets)]).permute(1, 0)

        # print(input_pos.shape)
        # input_pos = tensor([list(range(1, l.item() + 1))
        #                     + [0] * (self.max_len - l.item()) for l in input_len]).permute(1, 0)
        return self.position_encoding(input_pos) + self.seg_embeddings(seg_tsr)
