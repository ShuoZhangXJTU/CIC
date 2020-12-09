import torch
import copy
import torch.nn as nn
from transformers import BertModel, BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CorefBERT(nn.Module):
    def __init__(self):
        super(CorefBERT, self).__init__()
        name = 'bert-base-chinese' 
        self.tokenizer = BertTokenizer.from_pretrained(name)
        self.bert_model = BertModel.from_pretrained(name)
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def get_new_pos(self, raw_texts_ori, offsets):
        text_used = copy.deepcopy(raw_texts_ori)
        for bid, text_ in enumerate(text_used):
            text_.insert(offsets[bid], '[MASK]')
        input_lst = [''.join(x) for x in text_used]
        # -- be aware of truncation
        inputs_token = self.tokenizer(input_lst, add_special_tokens=True, padding='max_length',
                                      truncation=True, return_tensors="pt", max_length=300)
        pos_lst = [x.index(103) for x in inputs_token['input_ids'].tolist()]
        return pos_lst

    def forward(self, raw_texts, offsets):
        """
        -- raw_texts: list of list of words
        -- stc_lens: [[3,2,6...], [], ...]
        Return: doc_ttl_words * 1 * embedding_dim
        """
        pos_lst = self.get_new_pos(raw_texts, offsets)
        real_inputs = [''.join(x) for x in raw_texts]
        inputs_token = self.tokenizer(real_inputs, add_special_tokens=True, padding='max_length',
                                      truncation=True, return_tensors="pt", max_length=300)
        input_ids_tensor = inputs_token['input_ids'].to(device)
        segment_ids_tensor = inputs_token['token_type_ids'].to(device)
        input_mask_tensor = inputs_token['attention_mask'].to(device)
        hidden = self.bert_model(input_ids_tensor, input_mask_tensor, segment_ids_tensor)[0]

        # get pos - 1 & pos as h
        h_blk = []
        for tid, tsr in enumerate(torch.chunk(hidden, hidden.shape[0], dim=0)):
            gathered = torch.cat([tsr[:, pos_lst[tid] - 1, :], tsr[:, pos_lst[tid], :]], dim=1)
            h_blk.append(gathered.unsqueeze(1))

        return torch.cat(h_blk, dim=0)

