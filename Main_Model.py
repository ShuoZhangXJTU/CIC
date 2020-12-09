import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lstm_model import AttnLSTM, CenteredLSTM, BILSTM
from models.decoder_model import decoder_FFNN, BERT_HEAD
from models.transformer_model import Transformer, RelativePositionalEncoding
from models.BlankCoder import BlankCoder
from models.data_utils import insert_blank, select_blank
from models.bert_model import CorefBERT
from utils import check_zero_embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class main_model(nn.Module):
    def __init__(self, config, ntokens, nPOS):
        super(main_model, self).__init__()
        self.config = config
        self.device = device
        # -- embedding layer
        emsize = config["emsize"] if config["dataset"] == 'cn' else 200
        self.embedding = nn.Embedding(ntokens, emsize)

        if config["model"] == 'BILSTM':
            self.context_encoder = BILSTM(emsize, config["nhid_lstm"], config["nlayers_lstmEncoders"], config["dropout"])
            decoder_dim = config["nhid_lstm"] * 4

        if config["model"] == 'Transformer':
            self.context_encoder = Transformer(emsize, config["nlayers_attnEncoders"], config["dropout"],
                                                             config["nhead"], config["d_ffw"], 200)
            decoder_dim = emsize * 2

        if config["model"] == 'CenteredLSTM':
            self.blank_encoder = CenteredLSTM(emsize, config["nhid_lstm"], config["nlayers_lstmEncoders"], config["dropout"])
            decoder_dim = config["nhid_lstm"] * 4

        if config["model"] == 'AttnLSTM':
            self.blank_encoder = SegBiLSTM(emsize, config["nhid_lstm"], config["nlayers_lstmEncoders"],
                                                   config["dropout"], False)
            decoder_dim = config["nhid_lstm"] * 4

        if config["model"] == 'CorefBERT':
            self.context_encoder = CorefBERT()
            decoder_dim = 768 * 4

        if config["model"] == 'PBR':
            self.blank_encoder = BlankCoder(emsize, config["nlayers_attnEncoders"], config["nhead"], config["d_ffw"],
                                            config["k"], config["local_attn_dim"], config["N"],  config["dropout"])
            decoder_dim = emsize * 2  

        if self.config["cmp"]:
            decoder_dim = int(decoder_dim * 1.5)

        self.decoder = decoder_FFNN(decoder_dim, config["nhid_ffnn"], 2, config["dropout"])

    def forward_once(self, text_blk, raw_texts, texts, POS, offsets, stc_lens, sep_lst):
        if self.config["model"] == 'CorefBERT':
            b = self.context_encoder(raw_texts, offsets)

        if self.config["model"] == 'BILSTM' or self.config["model"] == 'Transformer':
            embedded = self.embedding(text_blk)

            stc_lens_new = [l + 1 if l < 200 else l for l in stc_lens]

            h_context = self.context_encoder(embedded, stc_lens_new, offsets)

            b = select_blank(h_context, offsets).permute(1, 0, 2)

        if self.config["model"] == 'AttnLSTM' or self.config["model"] == 'CenteredLSTM':
            embedded = self.embedding(texts)
            b = self.blank_encoder(embedded, offsets, stc_lens)

        if self.config["model"] == 'PBR':
            embedded = self.embedding(texts)
            b = self.blank_encoder(embedded, stc_lens, offsets, sep_lst)

        return b

    def forward(self, input_):

        text1_blk, text2_blk = input_['text1_blk'], input_['text2_blk']
        raw_text1, raw_text2 = input_['raw_text1'], input_['raw_text2']
        text1, text2 = input_['text1'], input_['text2']
        POS1, POS2 = input_['pos1'], input_['pos2']
        # -- tensor: Batch * 1
        offset1, offset2 = input_['offset1'], input_['offset2']
        stc_len1, stc_len2 = input_['stc_len1'], input_['stc_len2']
        sep_lst1, sep_lst2 = input_['sep_lst1'], input_['sep_lst2']

        b_1 = self.forward_once(text1_blk, raw_text1, text1, POS1, offset1, stc_len1, sep_lst1)
        b_2 = self.forward_once(text2_blk, raw_text2, text2, POS2, offset2, stc_len2, sep_lst2)

        if self.config["cmp"]:
            pair_tsr = torch.cat([b_1, b_2, torch.abs(b_1 - b_2)], dim=2)
        else:
            pair_tsr = torch.cat([b_1, b_2], dim=2)

        prob = F.softmax(self.decoder(pair_tsr).squeeze(1), dim=1)

        return b_1.squeeze(1), b_2.squeeze(1), prob.permute(1, 0)