class test_args():

    def __init__(self, batch_size=512, labeled_batch=0.75) -> None:
        
        self.batch_size = batch_size
        self.labeled_batch_size = round(batch_size*labeled_batch)
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