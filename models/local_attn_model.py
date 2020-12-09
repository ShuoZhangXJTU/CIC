import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StcAttention(nn.Module):
    def __init__(self, hidden_dim, use_wt):
        super(StcAttention, self).__init__()
        self.use_wt = use_wt
        attn_d = 128
        attn_r = 2
        self.attn_trans = nn.Sequential(
            nn.Linear(hidden_dim, attn_d, bias=False),
            nn.Tanh(),
            nn.Linear(attn_d, attn_r, bias=False),
            nn.Softmax(dim=1)
        )
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, h_context, attn_mask, char_lv_pos):
        # print(h_context.shape, attn_mask.shape)
        # h_context: blk_num * max_len * 768
        # -- attention
        # -- score: 1 * blk_num(seg_num) * max_len
        score = torch.mean(self.attn_trans(h_context).permute(2, 0, 1), dim=0, keepdim=True)
        # -- wt: blk_num * max_len
        wt = torch.tensor([[(1 / 2) ** (math.fabs(x - pos_) - 1)
                           for x in range(h_context.shape[1])]
                           for pos_ in char_lv_pos]).to(device)
        # print(wt.shape, len(char_lv_pos))
        if self.use_wt:
            score_pre = torch.mul(score, wt)

        # -- mask: blk_num * max_len
        score_mask = F.softmax(score_pre.masked_fill(attn_mask, -np.inf), dim=2)

        # (blk_num, 1, max_len) x (blk_num * max_len * 768) = blk_num, 1, hidden
        h_blank = torch.bmm(score_mask.permute(1, 0, 2), h_context)

        return h_blank


class SegAttention(nn.Module):
    def __init__(self, hidden_dim, use_wt):
        super(SegAttention, self).__init__()
        self.use_wt = use_wt
        attn_d = 128
        attn_r = 2
        self.pre_attn_trans = nn.Sequential(
            nn.Linear(hidden_dim, attn_d, bias=False),
            nn.Tanh(),
            nn.Linear(attn_d, attn_r, bias=False),
            nn.Softmax(dim=1)
        )
        self.fol_attn_trans = nn.Sequential(
            nn.Linear(hidden_dim, attn_d, bias=False),
            nn.Tanh(),
            nn.Linear(attn_d, attn_r, bias=False),
            nn.Softmax(dim=1)
        )
        self.reset_param()

    def reset_param(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, pre_emb_padded, fol_emb_padded,
                pre_attn_mask, fol_attn_mask,
                pre_is_valid_lst, fol_is_valid_lst):
        if pre_emb_padded is not None:
            # -- attention
            # -- score_pre: 2 * blk_num(seg_num) * max_pre_len
            score_pre = self.pre_attn_trans(pre_emb_padded.permute(1, 0, 2)).permute(2, 0, 1)
            # -- mask: seg_num * max_len
            score_pre_mask = F.softmax(score_pre.clone().masked_fill_(pre_attn_mask, -np.inf), dim=2)
            # -- (seg_num, 2, max_pre_len) x (seg_num, max_seg_len, hidden) = seg, 2, hidden
            h_pre = torch.bmm(score_pre_mask.permute(1, 0, 2), pre_emb_padded.permute(1, 0, 2))
            chunked_h_pre = torch.chunk(h_pre, h_pre.shape[0], dim=0)
            pre_idx, final_pre = 0, []
            for is_valid in pre_is_valid_lst:
                if is_valid:
                    final_pre.append(chunked_h_pre[pre_idx])
                    pre_idx += 1
                else:
                    final_pre.append(torch.zeros(1, 2, h_pre.shape[2]).to(device))
        else:
            final_pre = [torch.zeros(1, 2, fol_emb_padded.shape[2]).to(device)] * fol_emb_padded.shape[1]
        # --fol
        if fol_emb_padded is not None:
            score_fol = self.fol_attn_trans(fol_emb_padded.permute(1, 0, 2)).permute(2, 0, 1)
            score_fol_mask = F.softmax(score_fol.clone().masked_fill_(fol_attn_mask, -np.inf), dim=2)
            h_fol = torch.bmm(score_fol_mask.permute(1, 0, 2), fol_emb_padded.permute(1, 0, 2))
            chunked_h_fol = torch.chunk(h_fol, h_fol.shape[0], dim=0)
            fol_idx, final_fol = 0, []
            for is_valid in fol_is_valid_lst:
                if is_valid:
                    final_fol.append(chunked_h_fol[fol_idx])
                    fol_idx += 1
                else:
                    final_fol.append(torch.zeros(1, 2, h_fol.shape[2]).to(device))
        else:
            final_fol = [torch.zeros(1, 2, final_pre[0].shape[2]).to(device)] * len(final_pre)
        # seg * 1 * 2.hidden
        # print(torch.cat(final_pre, dim=0).shape)
        h_blank = torch.cat((torch.mean(torch.cat(final_pre, dim=0), dim=1, keepdim=True),
                             torch.mean(torch.cat(final_fol, dim=0), dim=1, keepdim=True)), dim=2)
        return h_blank
