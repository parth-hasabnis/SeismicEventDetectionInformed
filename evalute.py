# File to evalute the performance of trained Models

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
from Utility_files.metrics import metrics
from Utility_files.create_data import SeismicEventDataset
from Utility_files.utils import DatasetArgs, TestArguments
from models.CRNN import CRNN
# import librosa.display    # librosa can sometimes cause conflicts with matplotlib
from os.path import exists
from os import makedirs
import argparse

def test_model(weights, save_path, dataset_path, dataset_type, save_spectrograms: bool, plot_ROC: bool, save_metrics:bool, best_threshold=[0.5, 0.5], min_event_length=[1,2.5]):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    TEST_PATH = dataset_path    # Set dataset path 
    args = TestArguments()            # Use standard parameters

    if(not exists("Results" + save_path + "/Output Plots")):
        makedirs("Results" + save_path + "/Output Plots")
    if(not exists("Results" + save_path + "/Errors")):
        makedirs("Results" + save_path + "/Errors")


    f = open("Results" + save_path + "/training_args.json")  
    args = json.loads(f.read())
    dataset_kwargs = args["dataset_args"]
    training_args = args["training_args"]
    events = training_args["events"]
    f.close()
    dataset_kwargs["eval"] = True
    dataset_args = DatasetArgs(**dataset_kwargs)

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
    crnn.to(device)
    test_dataset = SeismicEventDataset(TEST_PATH, dataset_args, dataset_type)
    eval_loader = DataLoader(test_dataset, 256, drop_last=False)

    bases = np.linspace(-5, 5, 22)          # Thresholds for predictions
    thresholds = 1/(1 + np.exp(-bases))     # Generate exponentially distributed thresholds
    thresholds = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
    X_axis = ["False Alarm", "Miss", "Precison", "Recall"]
    error = list()
    error_values = np.zeros((len(thresholds),dataset_kwargs["num_events"], 4))

    for batch, (O, X,y) in enumerate(eval_loader):
        X = X.to(device)
        y = y.to(device)
        O = O.to(device)
        prediction, _ = crnn(X)

        if(plot_ROC or save_metrics):
            for i, threshold in enumerate(thresholds):
                threshold = np.ones(dataset_kwargs["num_events"])*threshold
                metric = metrics(prediction,y, threshold) 
                if batch == 0:
                    error.append(metric.Errors())
                else:
                    error[i] = error[i] + metric.Errors()

        if(save_spectrograms):
            print("Saving spectrograms")  
            metric = metrics(prediction,y, best_threshold)
            prediction = metric.get_thresholded_predictions(min_event_frames)
            heigths = [10, 20, 30, 40]
            upsampler = torch.nn.Upsample(scale_factor=pooling_time_ratio, mode='nearest')
            upsampler = upsampler.to(device)
            prediction  = prediction.permute(0, 2, 1)
            prediction = upsampler(prediction)
            y = y.permute(0, 2, 1)
            y = upsampler(y)
            X = X.permute(0, 1, 3, 2)
            O = O.permute(0, 1, 3, 2)

            for i in range(O.shape[0]):
                x = O[i].squeeze(dim=0)
                x = x.cpu().detach().numpy()
                # x = np.flip(x, axis=0)
                y_ = y[i].cpu().detach().numpy() - 0.5
                pred_ = prediction[i].cpu().detach().numpy() - 0.5
                sample_rate = 1000
                window_length_samples = int(round(sample_rate * dataset_kwargs["stft_window_seconds"]))
                hop_length_samples = int(round(sample_rate * dataset_kwargs["stft_hop_seconds"]))
                fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 9))
                # librosa.display.specshow(x,sr=sample_rate, n_fft=fft_length, win_length=window_length_samples,hop_length=hop_length_samples, x_axis='time', cmap = None, htk=False)
                ax.imshow(x)
                ax.plot(pred_[0]*heigths[0], linewidth=3)
                ax.plot(pred_[1]*heigths[1], linewidth=3)
                ax.plot(y_[0]*heigths[2], linewidth=3)
                ax.plot(y_[1]*heigths[3], linewidth=3)
                ax.set_ylim([0, dataset_kwargs["max_mel_band"]-1])
                ax.set_title("Spectrogram with predictions and targets")
                ax.set_xlabel("Time")
                ax.set_ylabel("Mel bands")
                ax.set_yticks(np.arange(0, dataset_kwargs["max_mel_band"], 2))
                plt.legend(["P Vehicle", "P Pedestrian", "T Vehicle", "T Pedestrian"])
                plt.savefig(f"Results/{save_path}/Output plots/{dataset_type}_{weights}_{batch}_{i}.png")  
                plt.close()
            print("Done Saving")
    error = np.array(error)
    if(save_metrics):
        for i, threshold in enumerate(thresholds):
            error_values[i, :, 0] = error[i, :, 1]/ (error[i, :, 1] + error[i, :, 0] + 0.0001)           # False Alarm
            error_values[i, :, 1] = error[i, :, 2]/ (error[i, :, 2] + error[i, :, 3] + 0.0001)           # Miss
            error_values[i, :, 2] = error[i, :, 3]/ (error[i, :, 3] + error[i, :, 1] + 0.0001)           # Precision
            error_values[i, :, 3] = error[i, :, 3]/ (error[i, :, 3] + error[i, :, 2] + 0.0001)           # Recall
            fig, ax = plt.subplots(nrows=1, ncols=dataset_kwargs["num_events"], figsize=(15,6))
            for axis in range(len(ax)):
                ax[axis].bar(X_axis,error_values[i,axis,:])
                ax[axis].set_xlabel("Metric")
                ax[axis].set_ylabel(f"Threshold = {threshold}")
                ax[axis].set_title(events[axis])
                ax[axis].set_yticks(np.linspace(0,1,11))
                plt.savefig(f"Results/{save_path}/Errors/{model_weights}_threshold_{threshold}.png")
            plt.close()

    if(plot_ROC):
        alpha = error_values[:,:,0]
        beta = 1-error_values[:,:,1]
        fig, ax = plt.subplots(1, dataset_kwargs["num_events"], figsize=(20,7))
        for axis in range(len(ax)):
            ax[axis].plot(alpha[:, axis], beta[:, axis])
            ax[axis].scatter(alpha[:, axis], beta[:, axis])
            for i, t in enumerate(thresholds):
                ax[axis].annotate(f"{t:.2f}", (alpha[i, axis] + 0.01, beta[i, axis] + 0.01))
            ax[axis].set_xlabel("Probability of False Alarm")
            ax[axis].set_ylabel("Probability of Detection")
            ax[axis].set_xlim([0,1])
            ax[axis].set_ylim([0,1])
            ax[axis].set_title(events[axis])
        fig.suptitle("Receiver Operating Characteristic")
        plt.savefig(f"Results/{save_path}/Errors/ROC_{model_weights}.png")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Evaluate a CRNN model for Seismic Event Detection")
    parser.add_argument("-f", "--file", help="Path for arguments file", default=".\default_eval_arguments.json")
    args = parser.parse_args()
    args_file = args.file
    with open(args_file) as f:
        data = json.load(f)

    eval_dataset_path = data["eval_dataset_path"]
    eval_dataset_type = data["eval_dataset_type"]
    model_weights = data["model_weights"]
    save_path = data["save_path"]
    best_threshold = data["best_threshold"]
    min_event_length = data["min_event_length"]

    test_model(model_weights, save_path, eval_dataset_path, eval_dataset_type, 
                save_spectrograms=False, plot_ROC=True, save_metrics=True,
                best_threshold=best_threshold, min_event_length=min_event_length)
