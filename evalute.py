# File to evalute the performance of trained Models

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import json
from Utility_files.metrics import StrongMetrics, WeakMetrics
from Utility_files.create_data import SeismicEventDataset
from Utility_files.utils import DatasetArgs, TestArguments
from models.CRNN import CRNN
# import librosa.display    # librosa can sometimes cause conflicts with matplotlib
from os.path import exists
from os import makedirs
import argparse

def test_model(weights, save_path, dataset_path, dataset_type, output_path,
               save_spectrograms: bool, plot_ROC: bool, save_metrics:bool, 
               best_threshold=[0.5, 0.5, 0.95], weak_threshold=[0.1, 0.1, 0.95], 
               min_event_length=[1,2.5], informed_thresh=True, bg_thresh = 0.95):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    TEST_PATH = dataset_path    # Set dataset path 
    args = TestArguments()            # Use standard parameters


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
    crnn.load_state_dict(torch.load(f"Results/{save_path}/Checkpoints/{weights}", map_location=device))
    crnn.eval()
    crnn.to(device)
    test_dataset = SeismicEventDataset(TEST_PATH, dataset_args, dataset_type)
    eval_loader = DataLoader(test_dataset, 256, drop_last=False)

    bases = np.linspace(-5, 5, 22)          # Thresholds for predictions
    thresholds = 1/(1 + np.exp(-bases))     # Generate exponentially distributed thresholds
    thresholds = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
    X_axis = ["False Alarm", "Miss", "Precison", "Recall", "F-score"]
    strong_error = list()
    strong_error_values = np.zeros((len(thresholds),dataset_kwargs["num_events"], len(X_axis)))
    weak_error = list()
    weak_error_values = np.zeros((len(thresholds),dataset_kwargs["num_events"], len(X_axis)))
    best_threshold_error_values = np.zeros((dataset_kwargs["num_events"], len(X_axis)))
    weak_best_threshold_error_values = np.zeros((dataset_kwargs["num_events"], len(X_axis)))
    weakAUC = np.zeros(dataset_kwargs["num_events"])
    strongAUC = np.zeros(dataset_kwargs["num_events"])

    for batch, (O, X,y) in enumerate(eval_loader):
        with torch.inference_mode():
            X = X.to(device)
            y = y.to(device)
            O = O.to(device)
            prediction, weak_pred = crnn(X)
            (weak_y, _) = torch.max(y, dim=1)

        if(plot_ROC or save_metrics):
            for i, threshold in enumerate(thresholds):
                threshold = np.ones(dataset_kwargs["num_events"])*threshold
                threshold[-1] = bg_thresh
                strong_metric = StrongMetrics(prediction,y, threshold, min_event_frames, informed_thresh=informed_thresh) 
                weak_metric = WeakMetrics(weak_pred, weak_y, threshold)
                if batch == 0:
                    strong_error.append(strong_metric.Errors())
                    weak_error.append(weak_metric.Errors())
                else:
                    strong_error[i] = strong_error[i] + strong_metric.Errors()
                    weak_error[i] = weak_error[i] + weak_metric.Errors()
        plot_metric = StrongMetrics(prediction,y, best_threshold, min_event_frames, informed_thresh=informed_thresh)
        plot_weak_metric = WeakMetrics(weak_pred, weak_y, weak_threshold)
        if batch == 0:
            best_threshold_error = plot_metric.Errors()
            weak_best_threshold_error = plot_weak_metric.Errors()
        else:
            best_threshold_error = best_threshold_error + plot_metric.Errors()
            weak_best_threshold_error = weak_best_threshold_error + plot_weak_metric.Errors()
        if(save_spectrograms):
            print("Saving spectrograms") 
            prediction = plot_metric.get_thresholded_predictions(min_event_frames)
            heigths = [10, 20, 30, 40, 50, 60]
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
                sample_rate = dataset_kwargs["sample_rate"]
                window_length_samples = int(round(sample_rate * dataset_kwargs["stft_window_seconds"]))
                hop_length_samples = int(round(sample_rate * dataset_kwargs["stft_hop_seconds"]))
                fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 9))
                plt.rcParams.update({'font.size': 20})
                # librosa.display.specshow(x,sr=sample_rate, n_fft=fft_length, win_length=window_length_samples,hop_length=hop_length_samples, x_axis='time', cmap = None, htk=False)
                ax.imshow(x)
                ax.set_xticks(np.linspace(0,100,11), np.linspace(0,10,11))
                ax.plot(pred_[0]*heigths[0], linewidth=5, color='magenta')
                ax.plot(pred_[1]*heigths[1], linewidth=5, color='orange')
                ax.plot(y_[0]*heigths[2], linewidth=5, color='cyan')
                ax.plot(y_[1]*heigths[3], linewidth=5, color='red')
                ax.set_ylim([0, dataset_kwargs["max_mel_band"]-1])
                ax.set_title("Spectrogram with predictions and targets")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Mel bands")
                ax.set_yticks(np.arange(0, dataset_kwargs["max_mel_band"], 4))
                ax.xaxis.set_major_locator(MultipleLocator(10))
                ax.xaxis.set_minor_locator(MultipleLocator(1))
                ax.yaxis.set_major_locator(MultipleLocator(4))
                ax.yaxis.set_minor_locator(MultipleLocator(1))

                ax1 = ax.twinx()
                ax1.set_ylim(0, 250)
                ax1.set_yticks(np.arange(25,sample_rate/2+1, 25))
                ax1.set_ylabel("Frequency (Hz)")
                ax1.yaxis.set_minor_locator(MultipleLocator(5))

                # ax.set_xticks(np.linspace(0,10,101))
                ax.legend(["Prediction Vehicle", "Prediction Pedestrian", "Target Vehicle", "Target Pedestrian"], fontsize="16")

                plt.savefig(f"Results/{output_path}/Output plots/{dataset_type}_{weights}_{batch}_{i}.png")  
                plt.close()
            print("Done Saving")
    strong_error = np.array(strong_error)
    weak_error = np.array(weak_error)
    if(plot_ROC or save_metrics):
        for i, threshold in enumerate(thresholds):

            # Strong prediction metrics
            strong_error_values[i, :, 0] = strong_error[i, :, 1]/ (strong_error[i, :, 1] + strong_error[i, :, 0] + 0.0001)           # False Alarm
            strong_error_values[i, :, 1] = strong_error[i, :, 2]/ (strong_error[i, :, 2] + strong_error[i, :, 3] + 0.0001)           # Miss
            strong_error_values[i, :, 2] = strong_error[i, :, 3]/ (strong_error[i, :, 3] + strong_error[i, :, 1] + 0.0001)           # Precision
            strong_error_values[i, :, 3] = strong_error[i, :, 3]/ (strong_error[i, :, 3] + strong_error[i, :, 2] + 0.0001)           # Recall
            # F-score
            strong_error_values[i, :, 4] = 2*(strong_error_values[i, :, 2] * strong_error_values[i, :, 3])/(strong_error_values[i, :, 2] + strong_error_values[i, :, 3] + 0.0001)

            # Weak prediction metrics
            weak_error_values[i, :, 0] = weak_error[i, :, 1]/ (weak_error[i, :, 1] + weak_error[i, :, 0] + 0.0001)           # False Alarm
            weak_error_values[i, :, 1] = weak_error[i, :, 2]/ (weak_error[i, :, 2] + weak_error[i, :, 3] + 0.0001)           # Miss
            weak_error_values[i, :, 2] = weak_error[i, :, 3]/ (weak_error[i, :, 3] + weak_error[i, :, 1] + 0.0001)           # Precision
            weak_error_values[i, :, 3] = weak_error[i, :, 3]/ (weak_error[i, :, 3] + weak_error[i, :, 2] + 0.0001)           # 
            # F-score
            weak_error_values[i, :, 4] = 2*(weak_error_values[i, :, 2] * weak_error_values[i, :, 3])/(weak_error_values[i, :, 2] + weak_error_values[i, :, 3] + 0.0001)

            1-strong_error_values[:,:,1]
            if i>0:
                strongAUC = strongAUC + abs((strong_error_values[i, :, 0] - strong_error_values[i-1, :, 0])*(2 - strong_error_values[i-1, :, 1] - strong_error_values[i, :, 1])/2)
                weakAUC = weakAUC + abs((weak_error_values[i, :, 0] - weak_error_values[i-1, :, 0])*(2 - weak_error_values[i-1, :, 1] - weak_error_values[i, :, 1])/2)
            
            if(save_metrics):
                plt.rcParams.update({'font.size': 16})
                fig, ax = plt.subplots(nrows=1, ncols=dataset_kwargs["num_events"]-1, figsize=(20,5))
                fig2, ax2 = plt.subplots(nrows=1, ncols=dataset_kwargs["num_events"]-1, figsize=(20,5))
                for axis in range(len(ax)):
                    ax[axis].bar(X_axis,strong_error_values[i,axis,:])
                    ax[axis].set_xlabel("Metric")
                    ax[axis].set_ylabel(f"Threshold = {threshold}")
                    ax[axis].set_title(events[axis])
                    ax[axis].set_yticks(np.linspace(0,1,11))
                    for metric_type in range(len(X_axis)):
                        val = "{:.2f}".format(strong_error_values[i, axis,metric_type]*100)
                        ax[axis].text(metric_type, strong_error_values[i, axis,metric_type], val, ha = 'center')
                    fig.savefig(f"Results/{output_path}/Errors/strong_{model_weights}_threshold_{threshold}.png")

                    ax2[axis].bar(X_axis,weak_error_values[i,axis,:])
                    ax2[axis].set_xlabel("Metric")
                    ax2[axis].set_ylabel(f"Threshold = {threshold}")
                    ax2[axis].set_title(events[axis])
                    ax2[axis].set_yticks(np.linspace(0,1,11))
                    for metric_type in range(len(X_axis)):
                        val = "{:.2f}".format(weak_error_values[i,axis,metric_type]*100)
                        ax2[axis].text(metric_type, weak_error_values[i,axis,metric_type], val, ha = 'center')
                    fig2.savefig(f"Results/{output_path}/Errors/weak_{model_weights}_threshold_{threshold}.png")
                plt.close(fig)
                plt.close(fig2)

        best_threshold_error_values[:,0] = best_threshold_error[:,1]/ (best_threshold_error[:,1] + best_threshold_error[:,0] + 0.0001)    # False Alarm
        best_threshold_error_values[:,1] = best_threshold_error[:,2]/ (best_threshold_error[:,2] + best_threshold_error[:,3] + 0.0001)    # Miss
        best_threshold_error_values[:,2] = best_threshold_error[:,3]/ (best_threshold_error[:,3] + best_threshold_error[:,1] + 0.0001)    # Precision
        best_threshold_error_values[:,3] = best_threshold_error[:,3]/ (best_threshold_error[:,3] + best_threshold_error[:,2] + 0.0001)    # Recall
        # F-score
        best_threshold_error_values[:, 4] = 2*(best_threshold_error_values[:, 2] * best_threshold_error_values[:, 3])/(best_threshold_error_values[:, 2] + best_threshold_error_values[:, 3] + 0.0001)


        weak_best_threshold_error_values[:,0] = weak_best_threshold_error[:,1]/ (weak_best_threshold_error[:,1] + weak_best_threshold_error[:,0] + 0.0001)    # False Alarm
        weak_best_threshold_error_values[:,1] = weak_best_threshold_error[:,2]/ (weak_best_threshold_error[:,2] + weak_best_threshold_error[:,3] + 0.0001)    # Miss
        weak_best_threshold_error_values[:,2] = weak_best_threshold_error[:,3]/ (weak_best_threshold_error[:,3] + weak_best_threshold_error[:,1] + 0.0001)    # Precision
        weak_best_threshold_error_values[:,3] = weak_best_threshold_error[:,3]/ (weak_best_threshold_error[:,3] + weak_best_threshold_error[:,2] + 0.0001)    # Recall
        # F-score
        weak_best_threshold_error_values[:, 4] = 2*(weak_best_threshold_error_values[:, 2] * weak_best_threshold_error_values[:, 3])/(weak_best_threshold_error_values[:, 2] + weak_best_threshold_error_values[:, 3] + 0.0001)
        if(1):
            plt.rcParams.update({'font.size': 16})
            fig, ax = plt.subplots(nrows=1, ncols=dataset_kwargs["num_events"]-1, figsize=(20,5))
            fig2, ax2 = plt.subplots(nrows=1, ncols=dataset_kwargs["num_events"]-1, figsize=(20,5))
            for axis in range(len(ax)):
                ax[axis].bar(X_axis,best_threshold_error_values[axis,:])
                ax[axis].set_xlabel("Metric")
                ax[axis].set_ylabel(f"Threshold = {best_threshold[axis]}")
                ax[axis].set_title(events[axis])
                ax[axis].set_yticks(np.linspace(0,1,11))
                for metric_type in range(len(X_axis)):
                    val = "{:.2f}".format(best_threshold_error_values[axis,metric_type]*100)
                    ax[axis].text(metric_type, best_threshold_error_values[axis,metric_type], val, ha = 'center')

                ax2[axis].bar(X_axis,weak_best_threshold_error_values[axis,:])
                ax2[axis].set_xlabel("Metric")
                ax2[axis].set_ylabel(f"Threshold = {weak_threshold[axis]}")
                ax2[axis].set_title(events[axis])
                ax2[axis].set_yticks(np.linspace(0,1,11))
                for metric_type in range(len(X_axis)):
                    val = "{:.2f}".format(weak_best_threshold_error_values[axis,metric_type]*100)
                    ax2[axis].text(metric_type, weak_best_threshold_error_values[axis,metric_type], val, ha = 'center')

            fig.savefig(f"Results/{output_path}/Errors/strong_{model_weights}_best_threshold.png")
            fig2.savefig(f"Results/{output_path}/Errors/weak_{model_weights}_best_threshold.png")
            plt.close(fig)
            plt.close(fig2)

    if(plot_ROC):
        plt.rcParams.update({'font.size': 12})
        alpha = strong_error_values[:,:,0]
        beta = 1-strong_error_values[:,:,1]
        fig, ax = plt.subplots(1, dataset_kwargs["num_events"]-1, figsize=(10,5))
        for axis in range(len(ax)):
            ax[axis].plot(alpha[:, axis], beta[:, axis])
            ax[axis].scatter(alpha[:, axis], beta[:, axis])
            # for i, t in enumerate(thresholds):
            #     ax[axis].annotate(f"{t:.2f}", (alpha[i, axis] + 0.01, beta[i, axis] + 0.01))
            ax[axis].set_xlabel("Probability of False Alarm")
            ax[axis].set_ylabel("Probability of Detection")
            ax[axis].set_xlim([0,1])
            ax[axis].set_ylim([0,1])
            title = f"{events[axis]}, AUC = {strongAUC[axis]:.2f}"
            ax[axis].set_title(title)
        fig.suptitle("Receiver Operating Characteristic")
        plt.savefig(f"Results/{output_path}/Errors/strong_ROC_{model_weights}.png")

        alpha = weak_error_values[:,:,0]
        beta = 1-weak_error_values[:,:,1]
        fig, ax = plt.subplots(1, dataset_kwargs["num_events"]-1, figsize=(10,5))
        for axis in range(len(ax)):
            ax[axis].plot(alpha[:, axis], beta[:, axis])
            ax[axis].scatter(alpha[:, axis], beta[:, axis])
            # for i, t in enumerate(thresholds):
            #     ax[axis].annotate(f"{t:.2f}", (alpha[i, axis] + 0.01, beta[i, axis] + 0.01))
            ax[axis].set_xlabel("Probability of False Alarm")
            ax[axis].set_ylabel("Probability of Detection")
            ax[axis].set_xlim([0,1])
            ax[axis].set_ylim([0,1])
            title = f"{events[axis]}, AUC = {weakAUC[axis]:.2f}"
            ax[axis].set_title(title)
        fig.suptitle("Receiver Operating Characteristic")
        plt.savefig(f"Results/{output_path}/Errors/weak_ROC_{model_weights}.png")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Evaluate a CRNN model for Seismic Event Detection")
    parser.add_argument("-f", "--file", help="Path for arguments file", default=".\default_eval_arguments.json")
    parser.add_argument("-r", "--roc", action="store_false", help="plot ROC. Default: True")
    parser.add_argument("-m", "--metrics", action="store_false", help="save metrics. Default: True")
    parser.add_argument("-p", "--plot", action="store_true", help="save output spectrograms. Default: False")
    args = parser.parse_args()
    args_file = args.file
    with open(args_file) as f:
        data = json.load(f)

    eval_dataset_path = data["eval_dataset_path"]
    eval_dataset_type = data["eval_dataset_type"]
    model_weights = data["model_weights"]
    save_path = data["save_path"]
    best_threshold = data["strong_threshold"]
    weak_threshold = data["weak_threshold"]
    min_event_length = data["min_event_length"]
    output_path = data["output_path"]
    informed_thresh = data["BIT"]

    if(not exists("Results" + output_path + "/Output Plots")):
        makedirs("Results" + output_path + "/Output Plots")
    if(not exists("Results" + output_path + "/Errors")):
        makedirs("Results" + output_path + "/Errors")

    output_args = "Results" + output_path + "\\eval_arguments.json"
    with open(output_args, 'w') as fp:
        json.dump(data, fp, indent=1)

    save_spectrograms = args.plot
    plot_ROC = args.roc
    save_metrics = args.metrics

    test_model(model_weights, save_path, eval_dataset_path, eval_dataset_type, output_path,
                save_spectrograms=save_spectrograms, plot_ROC=plot_ROC, save_metrics=save_metrics,
                best_threshold=best_threshold, weak_threshold=weak_threshold, min_event_length=min_event_length, informed_thresh=informed_thresh)
