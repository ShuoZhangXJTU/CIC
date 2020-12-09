import torch
import torch.nn as nn
import torch.nn.functional as F

class BERT_HEAD(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(BERT_HEAD, self).__init__()
        self.f = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_size, output_size)
        )
        self.reset_param()

    def reset_param(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, input):
        return self.f(input)


class decoder_FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(decoder_FFNN, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size)
        )
        self.reset_param()

    def reset_param(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, input):
        return self.f(input)
