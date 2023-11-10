from typing import Iterator
from torch.utils.data import IterableDataset, DataLoader
import os
import json
import obspy as opy
import numpy as np
import librosa 

class SeismicIterableDataset(IterableDataset):

    def __init__(self, file_path, args):
        self.file_path = file_path
        self.args = args
        self.index = 0

        st = opy.read(self.file_path)                       # Read Data file
        fs = round(1/st[0].stats.delta)                     # Read sampling rate of file
        if fs>self.args["sample_rate"]:         
            factor = round(fs/self.args["sample_rate"])     # Decimate file if fs>sample rate
            st.decimate(factor)
        
        self.data = st[0].data
        self.start = st[0].stats.starttime
        self.end = st[0].stats.endtime
    
    def __iter__(self):
        return iter(self.data)
        
    

# Test Iterable Dataset
if __name__ == "__main__":

    path1 = "D:\\Purdue\\Thesis\\eaps data\\Fall 23\\EAPS\\Raw Recordings\\NUS exit\\"
    fileID = "453001744..0.21.2020.04.26.11.00.00.000"
    cmp = 'Z'  
    file1Nm = path1 + fileID +'.'+ cmp + '.sac'

    args = {
        "sample_rate":500,
        "frame_len":10
    }

    dataset = SeismicIterableDataset(file1Nm, args)
    frame_size = args["sample_rate"]*args["frame_len"]
    frame_size = 5
    loader = DataLoader(dataset, batch_size=frame_size)

    for i, slice in enumerate(loader):
        print(slice)

        if i>5:
            break


        