import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.local_attn_model import SegAttention
from models.data_utils import get_offsets, get_stc_rep, get_seg_lst_by_blk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BILSTM(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout):
        super(BILSTM, self).__init__()
        self.bi_lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        for param in self.bi_lstm.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

    def forward(self, embedded, stc_lens, null_hold):
        # embedded: max_stc_len * batch * emsize
        embedded = self.dropout(embedded)
        chunked_embedded = torch.chunk(embedded, embedded.shape[1], dim=1)
        # reorder by stc_len
        sorted_stc = sorted([(len_, i, chunked_embedded[i]) for i, len_ in enumerate(stc_lens)], reverse=True)
        s_len, s_n, s_order = zip(*sorted_stc)
        s_len = list(s_len)
        # -- padded_embedded: max_stc_len * batch * emsize
        padded_embedded = torch.cat(list(s_order), dim=1)
        # do pack and pass RNN
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded_embedded, s_len)
        # pass RNN
        # input: seq_len, batch, input_size
        # output: seq_len, batch, num_directions * hidden_size
        rnn_sents, _ = self.bi_lstm(packed)
        # -- enc_sents: max_stc_len_by_s_len * batch * hidden_dim
        enc_sents, len_s = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents)
        # print(max(s_len))
        # reorder to normal
        # -- enc_chunked: list of max_stc_len * 1 * hidden_dim
        enc_chunked = torch.chunk(enc_sents, enc_sents.shape[1], dim=1)
        reordered = sorted([(list(s_n)[i], stc) for i, stc in enumerate(enc_chunked)])
        re_sn, re_emb = zip(*reordered)
        return torch.cat(list(re_emb), dim=1)


class AttnLSTM(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout, use_wt):
        super(AttnLSTM, self).__init__()
        self.fol_lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=0.5, bidirectional=True)
        self.pre_lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=0.5, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.seg_attn = SegAttention(hid_dim, use_wt)
        for param in self.fol_lstm.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
        for param in self.pre_lstm.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

    def forward(self, embedded, offsets, stc_lens):
        # -- embedded: max_stc_len * batch * emsize / 200, 128, 300

        embedded = self.dropout(embedded)
        pre_seg_lst, fol_seg_lst = get_seg_lst_by_blk(embedded, offsets, stc_lens)
        # -- find valid seg
        pre_is_valid_lst = [False if x is None else True for x in pre_seg_lst]
        fol_is_valid_lst = [False if x is None else True for x in fol_seg_lst]
        pre_seg_valid_lst, fol_seg_valid_lst = [], []
        for sid in range(len(pre_is_valid_lst)):
            if pre_is_valid_lst[sid]:
                pre_seg_valid_lst.append(pre_seg_lst[sid])
            if fol_is_valid_lst[sid]:
                fol_seg_valid_lst.append(fol_seg_lst[sid])
        # -- sort by len
        ori_len_pre = [emb_.shape[0] for emb_ in pre_seg_valid_lst]
        ori_len_fol = [emb_.shape[0] for emb_ in fol_seg_valid_lst]
        # -- for pre seg
        if len(ori_len_pre) != 0:
            sorted_pre_seg_valid = sorted([(emb_.shape[0], i, emb_)
                                           for i, emb_ in enumerate(pre_seg_valid_lst)], reverse=True)
            s_len_pre, s_n_pre, s_order_pre = zip(*sorted_pre_seg_valid)
            s_len_pre = list(s_len_pre)
            # -- pad
            padded_pre = [F.pad(emb_, (0, 0, 0, 0, 0, s_len_pre[0] - emb_.shape[0]), 'constant', 0.)
                          for emb_ in list(s_order_pre)]
            # -- pack
            # -- input T x B x * if not batch first
            packed_pre = torch.nn.utils.rnn.pack_padded_sequence(torch.cat(padded_pre, dim=1), s_len_pre)
            # -- pass RNN
            # -- input: seq_len, batch, input_size / output: seq_len, batch, num_directions * hidden_size
            rnn_sents_pre, _ = self.pre_lstm(packed_pre)
            # -- pad back
            # -- enc_sents: max_stc_len * stc_num * hidden_dim
            enc_sents_pre, len_s_pre = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents_pre)
            # -- order back
            chunked_pre = torch.chunk(enc_sents_pre, len(len_s_pre), dim=1)
            reordered_pre = sorted([(list(s_n_pre)[i], stc) for i, stc in enumerate(chunked_pre)])
            re_sn_pre, re_emb_pre = zip(*reordered_pre)
            # -- seg_len * seg_num(blk_num)  * hidden
            encoded_pre = torch.cat(list(re_emb_pre), dim=1)
            encoded_pre = encoded_pre.view(encoded_pre.shape[0],
                                       encoded_pre.shape[1],
                                       2,
                                       int(encoded_pre.shape[2] / 2))[:, :, 0, :]
            # -- generate seg_msk & seg wt
            pre_attn_mask = torch.tensor([len_ * [0] + (s_len_pre[0] - len_) * [1] for len_ in ori_len_pre]) \
                .bool().to(device)
        else:
            encoded_pre, pre_attn_mask = None, None

        if len(ori_len_fol) != 0:
            sorted_fol_seg_valid = sorted([(emb_.shape[0], i, emb_)
                                            for i, emb_ in enumerate(fol_seg_valid_lst)], reverse=True)
            s_len_fol, s_n_fol, s_order_fol = zip(*sorted_fol_seg_valid)
            s_len_fol = list(s_len_fol)
            padded_fol = [F.pad(emb_, (0, 0, 0, 0, 0, s_len_fol[0] - emb_.shape[0]), 'constant', 0.)
                          for emb_ in list(s_order_fol)]
            packed_fol = torch.nn.utils.rnn.pack_padded_sequence(torch.cat(padded_fol, dim=1), s_len_fol)
            rnn_sents_fol, _ = self.fol_lstm(packed_fol)
            enc_sents_fol, len_s_fol = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents_fol)
            chunked_fol = torch.chunk(enc_sents_fol, len(len_s_fol), dim=1)
            reordered_fol = sorted([(list(s_n_fol)[i], stc) for i, stc in enumerate(chunked_fol)])
            re_sn_fol, re_emb_fol = zip(*reordered_fol)
            encoded_fol = torch.cat(list(re_emb_fol), dim=1)
            encoded_fol = encoded_fol.view(encoded_fol.shape[0],
                                       encoded_fol.shape[1],
                                       2,
                                       int(encoded_fol.shape[2] / 2))[:, :, 1, :]
            fol_attn_mask = torch.tensor([len_ * [0] + (s_len_fol[0] - len_) * [1] for len_ in ori_len_fol])\
                .bool().to(device)
        else:
            encoded_fol, fol_attn_mask = None, None
        # print(encoded_fol.shape)
        h_blank = self.seg_attn(encoded_pre, encoded_fol,
                                pre_attn_mask, fol_attn_mask,
                                pre_is_valid_lst, fol_is_valid_lst)
        return h_blank


class CenteredLSTM(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout):
        super(CenteredLSTM, self).__init__()
        self.fol_lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=0.5, bidirectional=True)
        self.pre_lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=0.5, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        for param in self.fol_lstm.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
        for param in self.pre_lstm.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

    def forward(self, embedded, offsets, stc_lens):
        # -- embedded: max_stc_len * batch * emsize
        embedded = self.dropout(embedded)
        pre_seg_lst, fol_seg_lst = get_seg_lst_by_blk(embedded, offsets, stc_lens)
        # -- find valid seg
        pre_is_valid_lst = [False if x is None else True for x in pre_seg_lst]
        fol_is_valid_lst = [False if x is None else True for x in fol_seg_lst]
        pre_seg_valid_lst, fol_seg_valid_lst = [], []
        for sid in range(len(pre_is_valid_lst)):
            if pre_is_valid_lst[sid]:
                pre_seg_valid_lst.append(pre_seg_lst[sid])
            if fol_is_valid_lst[sid]:
                fol_seg_valid_lst.append(fol_seg_lst[sid])
        # -- sort by len
        ori_len_pre = [emb_.shape[0] for emb_ in pre_seg_valid_lst]
        ori_len_fol = [emb_.shape[0] for emb_ in fol_seg_valid_lst]
        if len(ori_len_pre) != 0:
            sorted_pre_seg_valid = sorted([(emb_.shape[0], i, emb_)
                                           for i, emb_ in enumerate(pre_seg_valid_lst)], reverse=True)
            s_len_pre, s_n_pre, s_order_pre = zip(*sorted_pre_seg_valid)
            s_len_pre = list(s_len_pre)
            # -- pad
            padded_pre = [F.pad(emb_, (0, 0, 0, 0, 0, s_len_pre[0] - emb_.shape[0]), 'constant', 0.)
                          for emb_ in list(s_order_pre)]
            # -- pack
            # -- input T x B x * if not batch first
            packed_pre = torch.nn.utils.rnn.pack_padded_sequence(torch.cat(padded_pre, dim=1), s_len_pre)
            # -- pass RNN
            # -- input: seq_len, batch, input_size / output: seq_len, batch, num_directions * hidden_size
            rnn_sents_pre, _ = self.pre_lstm(packed_pre)
            # -- pad back
            # -- enc_sents: max_stc_len * stc_num * hidden_dim
            enc_sents_pre, len_s_pre = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents_pre)
            # -- order back
            chunked_pre = torch.chunk(enc_sents_pre, len(len_s_pre), dim=1)
            reordered_pre = sorted([(list(s_n_pre)[i], stc) for i, stc in enumerate(chunked_pre)])
            re_sn_pre, re_emb_pre = zip(*reordered_pre)
            # -- seg_len * seg_num(blk_num) * hidden
            encoded_pre = torch.cat(list(re_emb_pre), dim=1)
            pre_dir = encoded_pre.view(encoded_pre.shape[0],
                                       encoded_pre.shape[1],
                                       2,
                                       int(encoded_pre.shape[2] / 2))[:, :, 0, :]
        else:
            pre_dir = None
        if len(ori_len_fol) != 0:
            sorted_fol_seg_valid = sorted([(emb_.shape[0], i, emb_)
                                           for i, emb_ in enumerate(fol_seg_valid_lst)], reverse=True)
            s_len_fol, s_n_fol, s_order_fol = zip(*sorted_fol_seg_valid)
            s_len_fol = list(s_len_fol)
            padded_fol = [F.pad(emb_, (0, 0, 0, 0, 0, s_len_fol[0] - emb_.shape[0]), 'constant', 0.)
                          for emb_ in list(s_order_fol)]
            packed_fol = torch.nn.utils.rnn.pack_padded_sequence(torch.cat(padded_fol, dim=1), s_len_fol)
            rnn_sents_fol, _ = self.fol_lstm(packed_fol)
            enc_sents_fol, len_s_fol = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents_fol)
            chunked_fol = torch.chunk(enc_sents_fol, len(len_s_fol), dim=1)
            reordered_fol = sorted([(list(s_n_fol)[i], stc) for i, stc in enumerate(chunked_fol)])
            re_sn_fol, re_emb_fol = zip(*reordered_fol)
            encoded_fol = torch.cat(list(re_emb_fol), dim=1)
            fol_rev = encoded_fol.view(encoded_fol.shape[0],
                                       encoded_fol.shape[1],
                                       2,
                                       int(encoded_fol.shape[2] / 2))[:, :, 1, :]
        else:
            fol_rev = None
        # -- do aggr here
        # -- get the last word as representation and use 0 vec if None
        pre_lst, fol_lst = [], []
        sel_pre, sel_fol = 0, 0
        emsize_ = pre_dir.shape[2] if pre_dir is not None else fol_rev.shape[2]
        for bid_pre, pre_is_valid in enumerate(pre_is_valid_lst):
            if pre_is_valid:
                idx_dim0, idx_dim1 = ori_len_pre[sel_pre] - 1, sel_pre
                pre_lst.append(pre_dir[idx_dim0, idx_dim1, :].view(1, 1, emsize_))
                sel_pre += 1
            else:
                pre_lst.append(torch.zeros(1, 1, emsize_).to(device))
        for bid_fol, fol_is_valid in enumerate(fol_is_valid_lst):
            if fol_is_valid:
                idx_dim0, idx_dim1 = 0, sel_fol
                fol_lst.append(fol_rev[idx_dim0, idx_dim1, :].view(1, 1, emsize_))
                sel_fol += 1
            else:
                fol_lst.append(torch.zeros(1, 1, emsize_).to(device))
        h_blank = torch.cat((torch.cat(pre_lst, dim=0), torch.cat(fol_lst, dim=0)), dim=2)
        return h_blank