import warnings

import torch.nn as nn
import torch

from models.RNN import BidirectionalGRU
from models.CNN import CNN


class CRNN(nn.Module):

    def __init__(self, n_in_channel, nclass, attention=False, activation="Relu", dropout=0,
                 train_cnn=True, rnn_type='BGRU', n_RNN_cell=64, n_layers_RNN=1, dropout_recurrent=0,
                 cnn_integration=False, **kwargs):
        super(CRNN, self).__init__()
        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration = cnn_integration
        n_in_cnn = n_in_channel

        if cnn_integration:
            n_in_cnn=1
        self.cnn = CNN(n_in_cnn, activation, dropout, **kwargs)
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requries_grad = False
        self.train_cnn = train_cnn
        nb_in = self.cnn.nb_filters[-1]
        if self.cnn_integration:
            self.fc = nn.Linear(nb_in * n_in_channel, nb_in)
            # nb_in = nb_in * n_in_channel
        self.rnn = BidirectionalGRU(nb_in,
                                    n_RNN_cell, dropout=dropout_recurrent, num_layers=n_layers_RNN)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell*2, nclass)
        self.sigmoid = nn.Sigmoid()
        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell*2, nclass)
            self.softmax = nn.Softmax(dim=-1)

    def load_cnn(self, state_dict):
        self.cnn.load_state_dict(state_dict)
        if not self.train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict["cnn"])
        if self.cnn_integration:
            self.fc.load_state_dict(state_dict["fc"])
        self.rnn.load_state_dict(state_dict["rnn"])
        self.dense.load_state_dict(state_dict["dense"])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "rnn": self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'dense': self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        if self.cnn_integration:
            state_dict["fc"] = self.fc.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return state_dict
    
    def forward(self, x):
        if self.cnn_integration:
            bs_in, nc_in = x.size(0), x.size(1)
            x = x.view(bs_in * nc_in, 1, *x.shape[2:])
        # print(f"Input shape={x.shape}")
        x = self.cnn(x)
        # print(f"CNN output shape={x.shape}")
        bs, chan, frames, freq = x.size()
        if self.cnn_integration:
            x = x.view(bs_in, chan * nc_in, frames, freq)
        
        if freq != 1:
            warnings.warn(f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq")
            #print("Problem - before permute")
            x = x.permute(0, 2, 1, 3)
            #print(f"CNN permute shape={x.shape}")
            #print("Problem - after permute")
            x = x.contiguous().view(bs, frames, chan * freq)
            #print("Problem - after view")
        else:
            #print("IN ELSE")
            x = x.squeeze(-1)
            #print("IN ELSE")
            x = x.permute(0, 2, 1)  # [bs, frames, chan]
            #print("IN ELSE")
        if self.cnn_integration:
            # print("CNN integration")
            x = self.fc(x)
            #print("CNN integration")

        # rnn features
        #print(f"PROBLEM - before RNN, shape={x.shape}")

        x = self.rnn(x)
    
        #print("PROBLEM - after RNN")
        x = self.dropout(x)
        #print("PROBLEM - after dropout")
        #print(f"Before Dense: {x.shape}")
        strong = self.dense(x)  # [bs, frames, nclass]
        #print(f"After Dense: {strong.shape}")
        #print("PROBLEM - after dense")
        strong = self.sigmoid(strong)
        # strong = strong.permute(0,2,1)
        """ if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(-2) / sof.sum(-2)   # [bs, nclass]
        else:
            weak = strong.mean(1) """
        return strong
    

if __name__ == '__main__':
    CRNN(64, 10, kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
         pooling=[(1, 4), (1, 4), (1, 4)])
