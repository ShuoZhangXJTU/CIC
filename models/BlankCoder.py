import numpy as np
import math, copy
import torch
import torch.nn as nn
from models.transformer_model import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BlankCoder(nn.Module):
    def __init__(self, emsize, nlayers_attnEncoders, nhead, d_ffw, k, local_attn_dim, N, dropout):
        super(BlankCoder, self).__init__()
        self.context_attn = Transformer(emsize, nlayers_attnEncoders, dropout, nhead, d_ffw, 200)
          
        self.pos_encoder = RelativePositionalEncoding(emsize, 200)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = torch.nn.LayerNorm(emsize)
        # -- blank encoding
        self.local_visible_pooling = LocalVisiblePooling(emsize, k, local_attn_dim)
        self.blank_updator = GlobalUpdate(emsize, N, nhead, dropout)

    def forward(self, embedded, stc_lens, offsets, sep_lst):
        
        H_bf = self.context_attn(embedded, stc_lens, offsets)
        b0_bf = self.local_visible_pooling(h_context, offsets, stc_lens, sep_lst)

        H = H_bf + self.pos_encoder(torch.tensor(stc_lens).to(device), offsets, 'H')
        H = self.dropout(self.LayerNorm(H))
        b0 = b0_bf + self.pos_encoder(torch.tensor([0]).to(device), offsets, 'b')
        b0 = self.dropout(self.LayerNorm(b0))

        b_c_attn_mask = torch.tensor([[0] * x + [1] * (200 - x)
                                              for x in stc_lens]).bool().to(device)
        b = self.blank_updator(b0, H, b_c_attn_mask)


class LocalVisiblePooling(nn.Module):
    def __init__(self, hidden_dim, K=2, attn_d=512):
        super(LocalVisiblePooling, self).__init__()
        self.K = K
        self.attn_trans = nn.Sequential(
            nn.Linear(hidden_dim, attn_d, bias=False),
            nn.Tanh(),
            nn.Linear(attn_d, 1, bias=False),
            nn.Softmax(dim=1)
        )
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, h_context, offsets, stc_lens, sep_lst, no_local):
        # h_context: max_len * blk_num * 768
        h_context_blk_lst = []
        h_context_chunked = torch.chunk(h_context, h_context.shape[1], dim=1)
        for stc_id in range(h_context.shape[1]):
            stc_len, pos, seps = stc_lens[stc_id], offsets[stc_id], sep_lst[stc_id]
            if len(seps) > 0:
                left_bound = 0
                for sep in seps:
                    if left_bound <= pos <= sep:
                        start_pos, end_pos = max(pos - self.K, left_bound), min(pos + self.K, sep)
                        sel_range = torch.tensor(list(range(start_pos, end_pos))).to(device)
                    left_bound = sep + 1
                if left_bound <= pos <= stc_len:
                    start_pos, end_pos = max(pos - self.K, left_bound), min(pos + self.K, stc_len)
                    sel_range = torch.tensor(list(range(start_pos, end_pos))).to(device)
            else:
                start_pos, end_pos = max(pos - self.K, 0), min(pos + self.K, stc_len)
                sel_range = torch.tensor(list(range(start_pos, end_pos))).to(device)
            # -- select local context
            if start_pos == end_pos:
                h_context_blk_lst.append(torch.zeros(1, 1, h_context.shape[2]).to(device))
            else:
                h_context_blk_lst.append(torch.index_select(h_context_chunked[stc_id], 0, sel_range))

        len_context_blk = [x.shape[0] for x in h_context_blk_lst]
        # - : 2K * batch * hidden
        h_context_blk = torch.cat([F.pad(x, (0, 0, 0, 0, 0, max(len_context_blk) - x.shape[0]), 'constant', 0.)
                                   for x in h_context_blk_lst], dim=1)
        # -- local context: generate mask
        # -- max_len * stc_num
        h_context_blk_mask = torch.tensor([[0] * x + [1] * (max(len_context_blk) - x)
                                           for x in len_context_blk]).permute(1, 0).bool().to(device)
        # -- max_len * stc_num
        score = torch.mean(self.attn_trans(h_context_blk), dim=2)
        score = F.softmax(score.masked_fill(h_context_blk_mask, -np.inf), dim=0).permute(1, 0)
        h_blank = torch.bmm(score.unsqueeze(1), h_context_blk.permute(1, 0, 2))
        return h_blank


class GlobalUpdate(nn.Module):
    def __init__(self, hidden_dim, N, head, dropout):
        super(GlobalUpdate, self).__init__()
        # -- Here we use the dynamic memory network to achieve semantic reasoning
        self.N = N
        self.LayerNorm = nn.LayerNorm(hidden_dim)
        self.gated_multi_head_attn = MultiHeadedAttention(head, hidden_dim, dropout)
        self.m_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, b0, H, b_c_attn_mask):
        b0 = b0.permute(1, 0, 2)
        b_t = b0.squeeze(0)

        for _ in range(self.N):
            b_t = b_t.unsqueeze(1)
            mask = mask.unsqueeze(1)
            m_t, gate = self.gated_multi_head_attn(b_t, H, H, b_c_attn_mask)
            m_t = m_t.squeeze(1)
            m_t = self.dropout(self.LayerNorm(m_t))
            b_t = self.m_cell(m_t, b_t)

        b_t = b_t.unsqueeze(1)

        return b_t


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def ScaledDotProductAttention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, -np.inf)

    p_attn = torch.sigmoid(scores)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % head == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // head
        self.h = head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        x, attn_rst = ScaledDotProductAttention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # return self.linears[-1](x), attn_rst
        return x, attn_rst




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

    def forward(self, input_len, offsets, type):
        """
          input_len: [BATCH_SIZE, 1]  with dim 1 as seq_len
        """
        if type == 'H':
            seg_tsr = torch.tensor([[0 if x < pos else 1 for x in range(200)] for pos in offsets]).permute(1, 0).to(device)
            tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
            # get input with padding
            input_pos = tensor([[(pos - x) if x < pos else (x + 1 - pos) for x in range(input_len[id_].item())]
             + [0]*(200-input_len[id_].item()) for id_, pos in enumerate(offsets)]).permute(1, 0)

            # print(input_pos.shape)
            # input_pos = tensor([list(range(1, l.item() + 1))
            #                     + [0] * (self.max_len - l.item()) for l in input_len]).permute(1, 0)
            return self.position_encoding(input_pos) + self.seg_embeddings(seg_tsr)
        else:
            return self.position_encoding(tensor([0]).to(device))

