import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import os
import json
import obspy as opy
import numpy as np
import librosa  

import itertools

class SeismicEventDataset(Dataset):
    """
    Seismic Event Dataset Class
    data_path: Path of Dataset files, each 10s long sampled at 500Hz
    args: 
    """

    def __init__(self, data_path, args, type):
        self.data_path = data_path
        self.args = args

        if type not in ['Synthetic', 'Unlabel']:
            raise Exception("Invalid Dataset type")

        self.type = type
        self.data = []
        self.labels = []
        
        for root, dirs, files in os.walk(self.data_path):
            for name in files:
                file = os.path.join(root, name)
                if(".sac" in file):
                    self.data.append(file)
                if(self.type == 'Synthetic'): 
                    if(".json" in file):
                        self.labels.append(file)

        print(f"Labels: {len(self.labels)}")
        print(f"Data: {len(self.data)}")
        if (len(self.data) == 0):
            raise Exception("Dataset is Empty")

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, idx):

        try:
            file = self.data[idx]
            st = opy.read(file)
        except:
            file = r"D:\Purdue\Thesis\eaps data\Fall 23\EAPS\DATASET\Train\Strong_Dataset_mini\Strong Labels\Real\Background\Background1.sac"
            st = opy.read(file)

        fs = round(1/st[0].stats.delta)
        if fs>self.args.sample_rate:
            factor = round(fs/self.args.sample_rate)
            st.decimate(factor)

        fs = round(1/st[0].stats.delta)
        sample_rate = fs
        data = st[0].data
        win_len = round(len(st[0].data)/fs)

        window_length_samples = int(round(sample_rate * self.args.stft_window_seconds))
        hop_length_samples = int(round(sample_rate * self.args.stft_hop_seconds))
        fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
        # n_mels = args.mel_bands
        num_spectrogram_bins = fft_length // 2 + 1

        spectrogram = librosa.feature.melspectrogram(y=data, sr=fs, n_fft=fft_length, win_length=window_length_samples,
                                                    hop_length=hop_length_samples, power=self.args.power, n_mels=self.args.mel_bands, htk=False)
        if self.args.eval:
            orig_spectrogram = spectrogram
            orig_spectrogram = np.log(orig_spectrogram + self.args.LOG_OFFSET)
            orig_spectrogram = orig_spectrogram[:self.args.max_mel_band,:]
        if self.args.mel_offset !=0:
            spectrogram[:self.args.mel_offset, :] = 0
        if self.args.normalize:
            spectrogram_norm = spectrogram/np.max(spectrogram)
            log_mel_spectrogram = np.log(spectrogram_norm + self.args.LOG_OFFSET)
        else:
            log_mel_spectrogram = np.log(spectrogram + self.args.LOG_OFFSET)
            
        log_mel_spectrogram = log_mel_spectrogram[:self.args.max_mel_band,:]
        if self.args.eval:
            assert np.all(log_mel_spectrogram.shape == orig_spectrogram.shape)

        len_data_s = len(data)/fs
        num_windows = log_mel_spectrogram.shape[1]
        scale = num_windows/len_data_s

        strong_label = torch.zeros(self.args.num_events, log_mel_spectrogram.shape[1])
        mask = torch.zeros(1, log_mel_spectrogram.shape[1])
        # if unlabel, label is a tensor of zeros

        if(self.type == 'Synthetic' and "Background" not in file):
            indexx = file.find("Data\\")
            skip = len("Data\\")
            root = file[:indexx]
            label_name = file[indexx+skip:] 
            label_name = label_name[:-4]
            label_path = root + "Labels\\" + label_name + ".json"
            f = open(label_path)
            label = json.load(f)
            f.close()

            key = list(label.keys())
            key = key[0]

            Order = ['traffic','pedestrian','background']  

            for stamp in label[key]:
                index = Order.index(stamp['Label'])
                start = round(stamp['Start']*scale)
                end = round(stamp['End']*scale)
                strong_label[index, start:end] = 1
                mask = mask + strong_label[index]
            temp = ~(mask.to(torch.bool))
            temp = temp.to(torch.float32)
            strong_label[-1, :] = temp

        downsampler = torch.nn.MaxPool1d(4,4)
        strong_label = downsampler(strong_label)
        strong_label = strong_label.T

        log_mel_spectrogram = log_mel_spectrogram.T      
        log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=0)  
        log_mel_spectrogram = log_mel_spectrogram.astype(np.float32)
        if self.args.eval:
            orig_spectrogram = orig_spectrogram.T      
            orig_spectrogram = np.expand_dims(orig_spectrogram, axis=0)  
            orig_spectrogram = orig_spectrogram.astype(np.float32)
            assert np.all(log_mel_spectrogram.shape == orig_spectrogram.shape)
            return orig_spectrogram, log_mel_spectrogram, strong_label
        else:
            return log_mel_spectrogram, strong_label
        

class TwoStreamBatchSampler(Sampler):

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size) -> None:
        # super(TwoStreamBatchSampler, self).__init__()

        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self): 
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_once(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size
    
class MultiStreamBatchSampler(Sampler):

    def __init__(self, classes:list, indices:list, batch_sizes:list) -> None:
        # super().__init__()

        self.classes = classes
        self.indices = indices
        self.class_batch_sizes = batch_sizes

        # assert len(classes) == len(indices) # == len(batch_sizes)
        self.class_num = len(classes)
        self.batch_size = sum(batch_sizes)

    def __iter__(self):
        iterators = []
        for i in range(len(self.class_batch_sizes)):
            iterators.append(grouper(iterate_once(self.indices[i]), self.class_batch_sizes[i]))
        return (sum(subbatch_ind, ()) for subbatch_ind in zip(*iterators))
    
    def __len__(self):
        val = np.inf
        for i in range(len(self.class_batch_sizes)):
            val = min(val, (len(self.indices[i]) // self.class_batch_sizes[i]))
        return val


def relabel_dataset(dataset):
    labeled_idxs = []
    unlabeled_idxs = []
    for idx, data in enumerate(dataset):
        label = data[1]
        label = torch.flatten(label)
        labelled = False
        for element in label:
            if torch.is_nonzero(element):
                labelled = True
                break
        if labelled:
            labeled_idxs.append(idx)
        elif not labelled:
            unlabeled_idxs.append(idx)

    return labeled_idxs, unlabeled_idxs

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
