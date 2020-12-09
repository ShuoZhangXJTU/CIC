import torch
import torch.utils.data as data
import torchtext
from cn_process import parse_jieba
from loading.imbalanced_sampler import ImbalancedDatasetSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def numerical(raw_POS, stoi):
    rst = []
    for POS in raw_POS:
        rst.append([stoi[x] for x in POS])
    return torch.tensor(rst)


def padding_numerical(raw_text, stoi, max_len):
    rst = []
    pad_i = stoi['<pad>']
    for text in raw_text:
        tmp = []
        for token in text:
            tmp.append(stoi[token])
        if len(text) < max_len:
            tmp += [pad_i] * (max_len - len(text))
        rst.append(tmp)
    return rst


def trans_tokenizer(text):
    return int(text)


def num_tokenizer(text):
    return [int(i) for i in text.split()] if text != 'None' else []


def tokenizer_doc(text):
    return text.split('|')

def tokenizer_kwd(text):
    text = text.split('|')
    rst = []
    for x in text:
        if x == 'null':
            rst.append([-1])
        else:
            rst.append([int(z) for z in x.split()])
    return rst

def pre_func(blk):
    return [x.split() for x in blk.split('|')]


def my_loader(config):
    # -- Fields
    INS_ID = torchtext.data.Field(sequential=True, use_vocab=False, tokenize=trans_tokenizer)
    if config["model"] == 'bert':
        TEXT = torchtext.data.Field(sequential=True, use_vocab=True, pad_token='<pad>', unk_token='[MASK]', fix_length=200)
    else:
        TEXT = torchtext.data.Field(sequential=True, use_vocab=True, pad_token='<pad>', unk_token='<blk>', fix_length=200)
    POS = torchtext.data.Field(sequential=True, use_vocab=True, pad_token='<pad>', fix_length=200)
    OFFSET = torchtext.data.Field(sequential=True, use_vocab=False, tokenize=trans_tokenizer)
    STCLEN = torchtext.data.Field(sequential=True, use_vocab=False, tokenize=trans_tokenizer)
    SEPLST = torchtext.data.Field(sequential=True, use_vocab=False, tokenize=num_tokenizer)
    LABEL = torchtext.data.Field(sequential=True, use_vocab=False, tokenize=trans_tokenizer)

    # -- load text data
    # -- Note: the dataset we load here concludes rows from ALL contracts
    if config["dataset"] == 'cn':
        filename_train = 'contracts_cn_train_0.csv'
        filename_validation = 'contracts_cn_test_0.csv'
        filename_test = 'contracts_cn_test_0.csv'
        word_emb_name = "sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5"
        pos_emb_name = ''
    elif config["dataset"] == 'en':
        filename_train = 'contracts_en_train_1.csv'
        filename_validation = 'contracts_en_test_1.csv'
        filename_test = 'contracts_en_test_1.csv'
        word_emb_name = "eng_word_embeddings"
        pos_emb_name = 'pos_tag_embeddings'

    train_dataset, val_dataset, test_dataset = torchtext.data.TabularDataset.splits(
        path=config["path_text_processed"], format='csv', skip_header=True,
        train=filename_train, validation=filename_validation, test=filename_test,
        fields=[('Unnamed: 0', INS_ID), ('text_blk1', TEXT), ('text_blk2', TEXT),
                ('text1', TEXT), ('text2', TEXT), ('offset1', OFFSET), ('offset2', OFFSET),
                ('stc_len1', STCLEN), ('stc_len2', STCLEN), ('POS1', POS),
                ('POS2', POS), ('sep_lst1', SEPLST), ('sep_lst2', SEPLST),
                ('label', LABEL)]
    )
    print('---- train ', filename_train,
          '\n---- dev   ', filename_validation,
          '\n---- test  ', filename_test)

    word_embedding = torchtext.vocab.Vectors(
        name=word_emb_name,
        cache=config["path_emb"]
    )
    TEXT.build_vocab(train_dataset, val_dataset, vectors=word_embedding)

    if config["use_POS"]:
        pos_embedding = torchtext.vocab.Vectors(
            name=pos_emb_name,
            cache=config["path_emb"]
        )
        POS.build_vocab(train_dataset, val_dataset, vectors=pos_embedding)
    else:
        POS.build_vocab(train_dataset, val_dataset)

    # -- build loader
    # -- custom defined loading func
    def col_fn(batch):
        """
        prepare batch
        --- for TEXT -> max_len * batch
        - cat stc in the doc
        - sort by doc_text len & store max_len
        - build doc_text order index
        --- for target -> catted 1D longTensor
        """
        text1_blk_raw, text2_blk_raw, text1_raw, text2_raw, POS1_raw, POS2_raw, offset1, offset2, \
        stc_len1, stc_len2, sep_lst1, sep_lst2, label = [x.text_blk1 for x in batch], [x.text_blk2 for x in batch],\
                                                        [x.text1 for x in batch], [x.text2 for x in batch], \
        [x.POS1 for x in batch], [x.POS2 for x in batch], [x.offset1 for x in batch], [x.offset2 for x in batch], \
        [x.stc_len1 for x in batch], [x.stc_len2 for x in batch], [x.sep_lst1 for x in batch], \
        [x.sep_lst2 for x in batch], [x.label for x in batch]
        # print(text1_blk_raw)
        text1_blk, text2_blk = TEXT.numericalize(TEXT.pad(text1_blk_raw), device), \
                               TEXT.numericalize(TEXT.pad(text2_blk_raw), device)
        text1, text2 = TEXT.numericalize(TEXT.pad(text1_raw), device), \
                       TEXT.numericalize(TEXT.pad(text2_raw), device)
        POS1, POS2 = POS.numericalize(POS.pad(POS1_raw), device), \
                     POS.numericalize(POS.pad(POS2_raw), device)
        return {'text1_blk': text1_blk, 'text2_blk': text2_blk,
                'raw_text1': text1_raw, 'raw_text2': text2_raw,
                'text1': text1, 'text2': text2,
                'pos1': POS1, 'pos2': POS2,
                'offset1': offset1, 'offset2': offset2,
                'stc_len1': stc_len1, 'stc_len2': stc_len2,
                'sep_lst1': sep_lst1, 'sep_lst2': sep_lst2, 'targets': torch.tensor(label).to(device)}

    ttl_batches = len(train_dataset.examples) // config["batch_size"]

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=config["batch_size"],
                                   collate_fn=col_fn, sampler=ImbalancedDatasetSampler(train_dataset))
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=config["batch_size"],
                                 collate_fn=col_fn, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=config["batch_size"],
                                  collate_fn=col_fn, shuffle=False)

    return train_loader, val_loader, test_loader, TEXT, ttl_batches, POS


def text2ipt(RAWINPUT, TEXT, POS):
    RAW1, RAW2 = RAWINPUT['t1'].split(), RAWINPUT['t2'].split()

    text1_blk_raw, text2_blk_raw = [RAW1], [RAW2]
    text1_raw, text2_raw = [[x for x in RAW1 if x != '<blk>']], [[x for x in RAW2 if x != '<blk>']]
    # print(RAW1)
    # print(text1_raw, text2_raw)
    offset1, offset2 = [RAW1.index('<blk>')], [RAW2.index('<blk>')]
    stc_len1, stc_len2 = [len(RAW1) - 1], [len(RAW2) - 1]
    sep_lst1, sep_lst2 = [[]], [[]]
    label = 1

    # print(stc_len1, stc_len2)

    text1_blk, text2_blk = TEXT.numericalize(TEXT.pad(text1_blk_raw), device), \
                           TEXT.numericalize(TEXT.pad(text2_blk_raw), device)

    text1, text2 = TEXT.numericalize(TEXT.pad(text1_raw), device), \
                   TEXT.numericalize(TEXT.pad(text2_raw), device)

    POS1, POS2 = None, None

    return {'text1_blk': text1_blk, 'text2_blk': text2_blk,
            'raw_text1': text1_raw, 'raw_text2': text2_raw,
            'text1': text1, 'text2': text2,
            'pos1': POS1, 'pos2': POS2,
            'offset1': offset1, 'offset2': offset2,
            'stc_len1': stc_len1, 'stc_len2': stc_len2,
            'sep_lst1': sep_lst1, 'sep_lst2': sep_lst2, 'targets': torch.tensor(label).to(device)}

