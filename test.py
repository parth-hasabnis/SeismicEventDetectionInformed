## Test Continuous Data

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from Utility_files.single_file_data import SeismicIterableDataset
from Utility_files.utils import simpleCount, multipleCount
from Utility_files.metrics import get_thresholded_predictions
from models.CRNN import CRNN
from os.path import exists
from os import makedirs
import argparse
import json
from os import path, walk
import glob
import datetime

def test_model(weights, save_path, dataset_path, file_format, output_path, save_spectrograms, time_zone="+00:00", best_threshold=[0.5, 0.5], min_event_length=[1,2.5]):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    TEST_PATHS = []    # Set dataset path 

    """for root, dirs, files in walk(dataset_path, topdown=False):
        for name in files:
            file = path.join(root, name)
            if(".sac" in file):
                TEST_PATHS.append(file)"""
    
    path_format = dataset_path + file_format
    TEST_PATHS = glob.glob(path_format)

    f = open("Results" + save_path + "/training_args.json")  
    args = json.loads(f.read())
    dataset_args = args["dataset_args"]
    training_args = args["training_args"]
    events = training_args["events"]
    f.close()

    f = open("Results" + save_path + "/crnn_args.json")
    crnn_kwargs = json.loads(f.read())
    f.close()
    pooling_time_ratio = 4  # 2 * 2
    min_event_frames = np.round(np.array(min_event_length)/pooling_time_ratio)
    batch_size_hour = 360         # 1 file = 10s, 360 files = 1 hour
    count_file_path = "Results" + output_path

    # Time Zone correction
    time_zone_list = time_zone.split(":")
    time_zone_correction = int(time_zone_list[0])*3600 + int(time_zone_list[1])*60  

    crnn = CRNN(**crnn_kwargs)
    crnn.load_state_dict(torch.load(f"Results/{save_path}/Checkpoints/{weights}", map_location=device))
    crnn.eval()
    crnn.to(device)
    for (file_num, file_path) in enumerate(TEST_PATHS):

        event_count_dict = {}
        test_dataset = SeismicIterableDataset(file_path, dataset_args)
        starttime = test_dataset.get_starttime() + time_zone_correction
        file_start = starttime.strftime("%A, %d. %B %Y %I:%M%p")
        event_count_dict["START"] = file_start

        test_loader = DataLoader(test_dataset, batch_size=batch_size_hour)
        for (index, O, X) in test_loader:
            with torch.inference_mode():
                X = X.to(device)
                O = O.to(device)
                prediction, _ = crnn(X)
            batch_start = index.cpu().detach().numpy()
            batch_start = int(batch_start[0])             # index of first file in batch, aka first file of hour
            # pred = prediction.cpu().detach().numpy()
            thresh_pred = get_thresholded_predictions(prediction, best_threshold, min_event_frames)
            thresh_pred = thresh_pred[:, :, :-1]            # Discard background predictions
            count = multipleCount(thresh_pred)              # Count per file
            new_count = np.sum(count, axis=0)               # All counts added ... count per hour
            hour_start = starttime + batch_start*10         # local starttime
            hour_end = starttime + batch_size_hour*10       # local endtime
            hour_start_string = hour_start.strftime("%A, %d. %B %Y %I:%M%p")    # startime string
            hour_end_string = hour_end.strftime("%A, %d. %B %Y %I:%M%p")        # endtime string

            try:
                assert hour_start.minute == 0
            except AssertionError:
                print(hour_start_string)

            event_count_dict[batch_start] = [hour_start_string, hour_end_string, new_count.tolist()]

            if save_spectrograms:
                heigths = [10, 20, 30, 40]
                upsampler = torch.nn.Upsample(scale_factor=pooling_time_ratio, mode='nearest')
                upsampler = upsampler.to(device)
                prediction = torch.from_numpy(thresh_pred)
                prediction  = prediction.permute(0, 2, 1)
                prediction = upsampler(prediction)
                X = X.permute(0, 1, 3, 2)
                O = O.permute(0, 1, 3, 2)

                for i in range(O.shape[0]):
                    x = O[i].squeeze(dim=0)
                    x = x.cpu().detach().numpy()
                    # x = np.flip(x, axis=0)
                    pred_ = prediction[i].cpu().detach().numpy()

                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 9))
                    ax.imshow(x)
                    ax.plot(pred_[0]*heigths[0], linewidth=3)
                    ax.plot(pred_[1]*heigths[1], linewidth=3)
                    ax.set_ylim([0, dataset_args["max_mel_band"]-1])
                    ax.set_title("Spectrogram with predictions and targets")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Mel bands")
                    ax.set_yticks(np.arange(0, dataset_args["max_mel_band"], 2))
                    plt.legend(["P Vehicle", "P Pedestrian"])
                    plt.savefig(f"Results/{output_path}/Output plots/{batch_start}_{i}.png")  
                    plt.close()

                    """print(pred_)

                    print("Index:", i)
                    key = input("Enter key: ")
                    if(key == "q"):
                        exit(0)
                    print("Shape:", pred_.shape)
                    if(i > 11):
                        exit(0)"""

          
        count_file = count_file_path + f"/count_{file_num}.json"
        with open(count_file, 'w') as fp:
            json.dump(event_count_dict, fp, indent=2)

    seed = 75
    torch.manual_seed(seed)
    np.random.seed(seed)

    f = open("Results" + save_path + "/crnn_args.json")
    crnn_kwargs = json.loads(f.read())
    f.close()
    pooling_time_ratio = 4  # 2 * 2
    min_event_frames = np.round(np.array(min_event_length)/pooling_time_ratio)

    crnn = CRNN(**crnn_kwargs)
    crnn.load_state_dict(torch.load(f"Results/{save_path}/Checkpoints/{weights}" ))
    crnn.eval()
    crnn.to(device)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Run a CRNN model for Seismic Event Detection for a continuous recording")
    parser.add_argument("-f", "--file", help="Path for arguments file", default=".\default_test_arguments.json")
    parser.add_argument("-p", "--plot", action="store_true", help="save output spectrograms. Default: False")
    args = parser.parse_args()
    args_file = args.file
    with open(args_file) as f:
        data = json.load(f)

    test_dataset_path = data["test_dataset_path"]
    model_weights = data["model_weights"]
    save_path = data["save_path"]
    best_threshold = data["best_threshold"]
    min_event_length = data["min_event_length"]
    output_path = data["output_path"]
    time_zone = data["time_zone"]
    file_format = data["format"]

    if(not exists("Results" + output_path + "/Output Plots")):
        makedirs("Results" + output_path + "/Output Plots")
    if(not exists("Results" + output_path + "/Errors")):
        makedirs("Results" + output_path + "/Errors")

    output_args = "Results" + output_path + "/test_arguments.json"
    with open(output_args, 'w') as fp:
        json.dump(data, fp, indent=2)

    save_spectrograms = args.plot

    test_model(model_weights, save_path, test_dataset_path, file_format, output_path, save_spectrograms,time_zone=time_zone, best_threshold=best_threshold, min_event_length=min_event_length)
