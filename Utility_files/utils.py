import torch
from torch import nn
import numpy as np

def weights_init(m):
    """ Initialize the weights of some layers of neural networks, here Conv2D, BatchNorm, GRU, Linear
        Based on the work of Xavier Glorot
    Args:
        m: the model to initialize
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('GRU') != -1:
        for weight in m.parameters():
            if len(weight.size()) > 1:
                nn.init.orthogonal_(weight.data)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    
class Arguments():
    def __init__(self, momentum, nesterov, epochs:int, consistency, batch_size=64, exclude_unlabelled=False,
                 labeled_batch_size=48, batch_sizes=[48, 16],consistency_type='kl', lr=0.01, initial_lr=0.005, lr_rampup = 7, ema_decay=0.999, 
                 consistency_rampup=5, early_stop=0.5, subsets=["synthetic", "unlabel"], weight_decay=0.999):
        super().__init__()

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.epochs = epochs
        self.consistency_type = consistency_type
        self.initial_lr = initial_lr
        self.lr_rampup = lr_rampup
        self.consistency = consistency
        self.ema_decay = ema_decay
        self.consistency_rampup = consistency_rampup
        self.early_stop = early_stop
        self.subsets = subsets
        self.exclude_unlabelled = exclude_unlabelled
        self.events = ['Vehicle', 'Pedestrian']

        self.batch_size = batch_size
        if exclude_unlabelled == True:           
            self.labeled_batch_size = batch_size
            self.batch_sizes = [self.labeled_batch_size]
        elif exclude_unlabelled == False:
            self.labeled_batch_size = labeled_batch_size
            assert batch_size == sum(batch_sizes)
            self.batch_sizes = batch_sizes
        
class DatasetArgs():
    """
    num_events: number of events in one file. Default=2
    LOG_OFFSET: offset to division by 0. Default = 0.001
    STFT Window seconds: seconds per window of STFT. Default=0.25
    STFT Hop seconds: seconds for hop = (window seconds - overlap seconds). Default=0.1
    Mel bands = number of mel bands. Default=64
    """

    def __init__(self, num_events=2, LOG_OFFSET = 0.0001, max_mel_band = 64, mel_offset=0,
                 stft_window_seconds = 0.25, stft_hop_seconds = 0.1, power = 2, normalize = False,
                 mel_bands = 128, sample_rate = 500) -> None:
        
        self.num_events = num_events
        self.LOG_OFFSET = LOG_OFFSET
        self.max_mel_band = max_mel_band # Max mel band of interest
        self.stft_window_seconds = stft_window_seconds # original 0.25
        self.stft_hop_seconds = stft_hop_seconds   # original 0.1
        self.mel_bands = mel_bands            # original 64
        self.sample_rate = sample_rate
        self.power = power
        self.normalize = normalize
        self.mel_offset = mel_offset

class TestArguments():

    def __init__(self, batch_size=512, labeled_batch=0.75) -> None:
        
        self.batch_size = batch_size
        # self.labeled_batch_size = round(batch_size*labeled_batch)
        # self.batch_sizes = [self.labeled_batch_size, batch_size - self.labeled_batch_size]
        self.events = ['Vehicle', 'Pedestrian']
        self.num_events = len(self.events)