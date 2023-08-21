import warnings

import torch
from torch import nn as nn

class BidirectionalGRU(nn.Module):

    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1) -> None:
        super(BidirectionalGRU, self).__init__()

        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)
    
    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent
    
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, droput=0, num_layers=1) -> None:
        super(BidirectionalLSTM, self).__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")
        
    def forward(self, input_feat):
        print("Inside RNN")
        recurrent, _ = self.rnn(input_feat)
        b, T, h = recurrent.size()
        t_rec = recurrent.contiguous().view(b * T, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(b, T, -1)
        return output