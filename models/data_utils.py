import torch
import math
import torch.nn.functional as F
from utils import list_unfold, tensor_insert

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def insert_blank(embedded, offsets, stc_lens, h_blank):
    # embedded: max_stc_len * batch * emsize / 200, 128, 300
    emb_lst = torch.chunk(embedded, embedded.shape[1], dim=1)
    blk_lst = torch.chunk(h_blank, h_blank.shape[0], dim=0)
    new_embedded = []
    for bid, emb_ in enumerate(emb_lst):
        new_embedded.append(tensor_insert(blk_lst[bid], offsets[bid], emb_, 0))
    return torch.cat(new_embedded, dim=1), [l + 1 if l < 200 else l for l in stc_lens]


def select_blank(h_context, offsets):
    rst = []
    for bid, emb_ in enumerate(torch.chunk(h_context, h_context.shape[1], dim=1)):
        rst.append(torch.index_select(emb_, 0, torch.tensor([offsets[bid]]).to(device)))
    return torch.cat(rst, dim=1)


def refine_prediction(output):
    pre_lst = []
    for chunked in torch.chunk(output, int(output.shape[0] ** 0.5) + 1):
        pre_tmp = chunked.gt(0).long().to(device)
        pre_lst.append(pre_tmp)
    return torch.cat(pre_lst).long().to(device)


def model_kwd(h_context, raw_texts, kwds):
    """
    return dict of  kwd: emb by averaging all kwds embs in file
    """
    kwd_emb_dict = dict.fromkeys(kwds[0], [])
    for pos, word in enumerate(raw_texts[0]):
        if word in kwd_emb_dict:
            kwd_emb_dict[word].append(h_context[pos, :, :])
    kwd2emb = dict.fromkeys(kwds[0], None)
    for kwd, emd_lst in kwd_emb_dict.items():
        kwd2emb[kwd] = torch.mean(torch.cat(emd_lst, dim=0), dim=0, keepdim=True)
    return kwd2emb


# : segment operation ----------------------------------------------------
def get_seg_lst_by_blk(embedded, s_offsets, stc_lens):
    """
    :param embedded: max_len, batch, emsize
    :param s_offsets: list of offset
    :param stc_lens: list of stc_len
    :return: pre/fol_lst: list with element being tensor(len_pre/fol, 1, dim) or None
    """
    # -- chunked_embedded: list of max_len, 1, dim
    chunked_embedded = torch.chunk(embedded, embedded.shape[1], dim=1)
    pre_lst, fol_lst = [], []
    for batch_id, emb in enumerate(chunked_embedded):
        # print(s_offsets[batch_id], stc_lens[batch_id])
        tmp_pre = emb[:s_offsets[batch_id], :, :] if s_offsets[batch_id] != 0 else None
        tmp_fol = emb[s_offsets[batch_id]:stc_lens[batch_id], :, :] \
            if s_offsets[batch_id] != stc_lens[batch_id] else None
        pre_lst.append(tmp_pre)
        fol_lst.append(tmp_fol)
    return pre_lst, fol_lst


def get_bid2sid(offsets, stc_lens):
    """
    :param offsets: list of 1 tensor, offsets by doc
    :param stc_lens: list of 1 list, stc length
    :return: bid2sid: dict of int, blank id 2 stc id, both start from 0
    """
    bid2sid, acc_len = dict(), 0
    sid = 0
    for bid, offset in enumerate(offsets[0]):
        while offset >= acc_len + stc_lens[0][sid]:
            acc_len += stc_lens[0][sid]
            sid += 1
        if acc_len <= offset < acc_len + stc_lens[0][sid]:
            bid2sid[bid] = sid
    return bid2sid


def get_offsets(offsets, stc_lens):
    """
    calculate intra_stc offsets for blanks
    :param offsets: list of 1 tensor, offsets by doc
    :param stc_lens: list of 1 list, stc length
    :return: s_offsets: dict of lists, sid 2 intra-stc offset list
    """
    s_offsets, acc_len = dict(), 0
    stc_id, tmp = 0, []
    for offset in offsets[0]:
        while offset >= acc_len + stc_lens[0][stc_id]:
            s_offsets[stc_id] = tmp
            tmp = []
            acc_len += stc_lens[0][stc_id]
            stc_id += 1
        if acc_len <= offset < acc_len + stc_lens[0][stc_id]:
            tmp.append(offset.item() - acc_len)
    s_offsets[stc_id] = tmp
    while len(s_offsets) < len(stc_lens[0]):
        stc_id += 1
        s_offsets[stc_id] = []
    
    return s_offsets


def get_stc_rep(hidden, stc_lens):
    """
    :param hidden: total_words x 1 x hidden
    :param stc_lens: list of 1 list, stc length
    :return: stcs: list of tensors cut by stc length
    """
    stcs, s_pos = [], 0
    for length in stc_lens[0]:
        stc_range = torch.tensor(list(range(s_pos, s_pos + length))).to(device)
        stcs.append(torch.index_select(hidden, 0, stc_range.long()))
        s_pos += length
    return stcs


def get_sid2pureContext(offsets_in_stc_dict, stc_emb_lst):
    sid2pureContext = dict()
    for sid, i_offset_lst in offsets_in_stc_dict.items():
        stc_emb = stc_emb_lst[sid]
        if len(i_offset_lst) == 0:
            sid2pureContext[sid] = stc_emb
            continue
        # -- remove blk-emb in stc
        start_pos_, refined_stc_seg_lst = 0, []
        for i_offset in i_offset_lst:
            if start_pos_ == i_offset:
                start_pos_ += 1
                continue
            refined_stc_seg_lst.append(stc_emb[start_pos_:i_offset, :, :])
            start_pos_ = i_offset + 1
        refined_stc_seg_lst.append(stc_emb[start_pos_:, :, :])
        sid2pureContext[sid] = torch.cat(refined_stc_seg_lst, dim=0)
    return sid2pureContext
