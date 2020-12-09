import torch
import time
import math
import spacy
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
import operator
from functools import reduce
from scipy.linalg import norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_zero_embeddings(embedded, stc_lens):
    # embedded: max_stc_len * batch * emsize / 200, 128, 300
    zero_percentage = []
    for bid, emb_ in enumerate(torch.chunk(embedded, embedded.shape[1], dim=1)):
        emb_ = emb_.squeeze(1)
        emb_ = torch.index_select(emb_, 0, torch.tensor(list(range(stc_lens[bid]))).to(device))
        zero_num = 0
        for i in range(emb_.shape[0]):
            if emb_[i].sum().item() == 0:
                zero_num += 1
        zero_percentage.append(float(zero_num)/emb_.shape[0])
    return zero_percentage


# : data operations --------------------------------------------------------------------
def tensor_insert(row, pos, tensor, dim):
    if pos == 0:
        return torch.cat((row, tensor), dim).to(device)
    if pos == tensor.shape[dim]:
        return torch.cat((tensor, row), dim).to(device)
    first_half = torch.index_select(tensor, dim,
                                    torch.tensor(list(range(pos))).long().to(device))
    # -1 is for max_len = 200
    second_half = torch.index_select(tensor, dim,
                                     torch.tensor(list(range(pos, tensor.shape[dim]))).to(device))
    return torch.cat((first_half, row, second_half), dim).to(device)


def list_unfold(lst):
    return reduce(operator.add, lst)


# : pre-processing models ----------------------------------------------------------------------------
def spacy_process():
    nlp = spacy.load('zh')
    doc = nlp('本合同以中文签署一式blk份，买方持blk份，卖方持blk份，具有相同法律效力')
    # displacy.serve(doc, style="dep")
    for token in doc:
        if token.text == 'blk':
            print([x.text for x in token.ancestors])
            print(token.children)
        # print(token.text, token.dep_, token.head)
    print('---')
    doc = nlp('本合同以中文签署一式blk份，买方持blk份，卖方持blk份，具有相同法律效力')
    for token in doc:
        print(token.text, token.dep_, token.head)


# : print funcs -----------------------------------------------------------------------------
def print_progress(start_time, percent=0, width=30):
    percent *= 100
    left = int(width * percent // 100)
    right = width - left
    print('\r[', '\u2588' * left, ' ' * right, ']',
          f' {percent:.2f}%', f' {(time.time() - start_time):.0f}s',
          sep='', end='', flush=True)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def print_nxG(nx_G, title):
    # print(nx_G.edges)
    fig = plt.figure(figsize=(24, 24))
    ax = plt.subplot(111)
    ax.set_title(title, fontsize=10)
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw_spectral(nx_G, with_labels=True, node_color=[[.7, .7, .7]])
    plt.tight_layout()
    plt.savefig("{}.png".format(title))
    print('pic_saved')


# : stc similarity features --------------------------------------------------------------------------
def calc_text_pair_features(all_text, stc_lens):
    IDF = gen_idf(all_text[0], stc_lens)
    s_position, stc_text = 0, []
    for s_len in stc_lens[0]:
        stc_text.append(all_text[0][s_position: s_position+s_len])
        s_position += s_len
    sid2tf, sid2tfidf, sid2set = dict(), dict(), dict()
    for sid, text in enumerate(stc_text):
        sid2tf[sid] = gen_tf(text)
        sid2tfidf[sid] = gen_tfidf(text, IDF)
        sid2set[sid] = set(text)
        if '<blk>' in sid2tf[sid]:
            del sid2tf[sid]['<blk>']
        if '<blk>' in sid2tfidf[sid]:
            del sid2tfidf[sid]['<blk>']
        if '<blk>' in sid2set[sid]:
            sid2set[sid].remove('<blk>')

    features = []
    for sid1 in range(len(stc_text)):
        for sid2 in range(len(stc_text)):
            str1, str2 = sid2set[sid1], sid2set[sid2]
            if len(str1) == 0 or len(str2) == 0:
                jaccard_common_words, ochiai_common_words = 0.0, 0.0
            else:
                jaccard_common_words = float(len(str1 & str2)) / len(str1 | str2)
                ochiai_common_words = float(len(str1 & str2)) / math.sqrt(len(str1) * len(str2))
            features.append([cosine_sim(sid2tf[sid1], sid2tf[sid2]),
                             cosine_sim(sid2tfidf[sid1], sid2tfidf[sid2]),
                             jaccard_common_words,
                             ochiai_common_words])
    return features


def cosine_sim(a, b):
    if len(b) < len(a):
        a, b = b, a
    res = 0
    for key, a_value in a.items():
        res += a_value * b.get(key, 0)
    if res == 0:
        return 0
    try:
        res = res / (norm(list(a.values())) * norm(list(b.values())))
    except ZeroDivisionError:
        res = 0
    return res


def gen_tf(tokens):
    """
    Given a segmented string, return a dict of tf.
    """
    # tokens = text.split()
    total = len(tokens)
    tf_dict = {}
    for w in tokens:
        tf_dict[w] = tf_dict.get(w, 0.0) + 1.0
    for k in tf_dict:
        tf_dict[k] /= total
    return tf_dict


def gen_idf(all_text, stc_lens):
    total_stcs = len(stc_lens[0])
    idf_dict = {}
    start_pos = 0
    for s_len in stc_lens[0]:
        for w in list(set(all_text[start_pos:start_pos+s_len])):
            idf_dict[w] = idf_dict.get(w, 0.0) + 1.0
        start_pos += s_len
    for k in idf_dict:
        idf_dict[k] /= total_stcs
    return idf_dict


def gen_tfidf(tokens, idf_dict):
    """
    Given a segmented string and idf dict, return a dict of tfidf.
    """
    # tokens = text.split()
    total = len(tokens)
    tfidf_dict = {}
    for w in tokens:
        tfidf_dict[w] = tfidf_dict.get(w, 0.0) + 1.0
    for k in tfidf_dict:
        tfidf_dict[k] *= idf_dict.get(k, 0.0) / total
    return tfidf_dict


if __name__ == '__main__':
    spacy_process()
