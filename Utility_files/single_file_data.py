from torch.utils.data import IterableDataset, DataLoader
import os.path
import json
import obspy as opy
import numpy as np
import librosa 
from scipy import signal

class SeismicIterableDataset(IterableDataset):

    def __init__(self, file_path, args):
        self.file_path = file_path
        self.args = args
        self.index = 0

        self.args["frame_len"] = 10

        st = opy.read(self.file_path)                       # Read Data file
        fs = round(1/st[0].stats.delta)                     # Read sampling rate of file
        if fs>self.args["sample_rate"]:         
            factor = round(fs/self.args["sample_rate"])     # Decimate file if fs>sample rate
            st.decimate(factor)
        fs = round(1/st[0].stats.delta)
        assert fs == self.args["sample_rate"]
        
        self.st = st.copy()
        self.data = st[0].data
        self.start = st[0].stats.starttime
        if self.start.minute != 0:
            self. start = self.start.replace(second=0, microsecond=0, minute=0, hour=self.start.hour+1)

        self.window_length_samples = int(round(self.args["sample_rate"] * self.args["stft_window_seconds"]))
        self.hop_length_samples = int(round(self.args["sample_rate"] * self.args["stft_hop_seconds"]))
        self.fft_length = 2 ** int(np.ceil(np.log(self.window_length_samples) / np.log(2.0)))

        self.num_frames = round(len(self.data)/(fs*self.args["frame_len"]))
        display_start = self.start.strftime("%A, %d. %B %Y %I:%M%p")            # Time to display in console
        print(f"File Start: {display_start}. Number of frames in .sac file: {self.num_frames}")

    def get_starttime(self):
        """
        Return start time of data file
        """
        return self.start

    def get_slice(self):
        win_bgn = self.start + self.index*self.args["frame_len"]
        win_end = win_bgn+self.args["frame_len"]
        st_slice = self.st.slice(win_bgn, win_end)
        self.index = self.index + 1
        data = st_slice[0].data
        return self.index-1, data

    def get_spectrogram(self, data):
        spectrogram = librosa.feature.melspectrogram(y=data, sr=self.args["sample_rate"], n_fft=self.fft_length, win_length=self.window_length_samples,window=signal.windows.hann,
                                                    hop_length=self.hop_length_samples, power=self.args["power"], n_mels=self.args["mel_bands"], htk=False)
        
        orig_spectrogram = spectrogram                                              # Orig spectrogram = not zeroed out below mel offset
        orig_spectrogram = np.log(orig_spectrogram + self.args["LOG_OFFSET"])
        orig_spectrogram = orig_spectrogram[:self.args["max_mel_band"],:]
        if self.args["mel_offset"] !=0:
            spectrogram[:self.args["mel_offset"], :] = 0
        if self.args["normalize"]:
            spectrogram_norm = spectrogram/np.max(spectrogram)
            log_mel_spectrogram = np.log(spectrogram_norm + self.args["LOG_OFFSET"])
        else:
            log_mel_spectrogram = np.log(spectrogram + self.args["LOG_OFFSET"]) 
        log_mel_spectrogram = log_mel_spectrogram[:self.args["max_mel_band"],:]

        assert np.all(log_mel_spectrogram.shape == orig_spectrogram.shape)

        log_mel_spectrogram = log_mel_spectrogram.T      
        log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=0)  
        log_mel_spectrogram = log_mel_spectrogram.astype(np.float32)

        orig_spectrogram = orig_spectrogram.T      
        orig_spectrogram = np.expand_dims(orig_spectrogram, axis=0)  
        orig_spectrogram = orig_spectrogram.astype(np.float32)

        assert np.all(log_mel_spectrogram.shape == orig_spectrogram.shape)
        return orig_spectrogram, log_mel_spectrogram
         
    def __iter__(self):
        index = 0
        while index<self.num_frames-1:
            index, data = self.get_slice()
            o, x = self.get_spectrogram(data)
            if (o.shape != (1, 101, 64)):
                break
            yield index, o, x


# Test Iterable Dataset
if __name__ == "__main__":

    path1 = "D:\\Purdue\\Thesis\\eaps data\\Fall 23\\EAPS\\Raw Recordings\\NUS exit\\"
    fileID = "453001744..0.21.2020.04.26.11.00.00.000"
    cmp = 'Z'  
    file1Nm = path1 + fileID +'.'+ cmp + '.sac'

    file = open(os.path.dirname(__file__) + '/../default_arguments.json')
    args = json.loads(file.read())
    file.close
    
    dataset = SeismicIterableDataset(file1Nm, args["dataset_args"])
    loader = DataLoader(dataset, batch_size=200)

    for i,(o, x) in enumerate(loader):
        print(o.shape, x.shape)

        if i>5:
            break


        