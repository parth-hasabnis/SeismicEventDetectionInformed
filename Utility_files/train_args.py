class train_args():

    def __init__(self, momentum, weight_decay, nesterov, epochs:int, 
                 consistency, exclude_unlabeled:bool, batch_size=512,
                 labeled_batch=0.75, subsets=['synthetic', 'unlabel'],
                 consistency_type='kl', lr=0.01,initial_lr=0.005, 
                 lr_rampup=7, ema_decay=0.99, consistency_rampup=5) -> None:
        
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
        self.labeled_batch_size = round(batch_size*labeled_batch)
        self.exclude_unlabeled = exclude_unlabeled
        self.batch_size = batch_size
        self.consistency_rampup = consistency_rampup
        self.subsets = subsets
        self.batch_sizes = [self.labeled_batch_size, batch_size - self.labeled_batch_size]
        self.events = ['Vehicle', 'Pedestrian']

        # Spectrogram parameters
        self.num_events = 2             # Events 
        self.LOG_OFFSET = 0.001         
        self.max_mel_band = 64          # Max mel band of interest
        self.stft_window_seconds = 0.25 # original 0.25
        self.stft_hop_seconds = 0.1     # original 0.1
        self.mel_min_hz = 1
        self.mel_bands = 128            # original 64
        self.mel_max_hz = 500
