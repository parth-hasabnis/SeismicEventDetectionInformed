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
import librosa
import librosa.display

def test_model(weights, save_path, dataset_path, dataset_type, save_spectrograms: bool, plot_ROC: bool, save_metrics:bool, best_threshold=[0.5, 0.5]):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_PATH = dataset_path    # Set dataset path 
    args = TestArguments()            # Use standard parameters
    dataset_args = DatasetArgs()

    seed = 75
    torch.manual_seed(seed)
    np.random.seed(seed)

    f = open("Results" + save_path + "/crnnn_args.json")
    crnn_kwargs = json.loads(f.read())
    f.close()
    pooling_time_ratio = 4  # 2 * 2

    crnn = CRNN(**crnn_kwargs)
    crnn.load_state_dict(torch.load(f"Results/{save_path}/Checkpoints/{weights}" ))
    crnn.to(device)
    test_dataset = SeismicEventDataset(TEST_PATH, dataset_args, dataset_type)
    eval_loader = DataLoader(test_dataset, args.batch_size, drop_last=False)

    bases = np.linspace(-5, 5, 22)          # Thresholds for predictions
    thresholds = 1/(1 + np.exp(-bases))     # Generate exponentially distributed thresholds

    if(plot_ROC or save_metrics):
        yConsolidated = []
        predictionConsolidated = []    

    for batch, (X,y) in enumerate(eval_loader):
        X = X.to(device)
        y = y.to(device)
        prediction, _ = crnn(X)
    alpha = []
    beta = []

    if(plot_ROC or save_metrics):
        for i, threshold in enumerate(thresholds):
            threshold = np.ones(args.num_events)*threshold
            metric = metrics(prediction,y, threshold) 
            errors = metric.Errors()
            alpha.append(np.array(errors["type 1"]))
            beta.append(1 - np.array(errors["type 2"]))
            error = metric.Errors()
            error_values = np.array(list(error.values()))

            if(save_metrics):
                fig, ax = plt.subplots(nrows=1, ncols=args.num_events, figsize=(15,6))
                for axis in range(len(ax)):
                    ax[axis].bar(list(error.keys()),error_values[:,axis])
                    ax[axis].set_xlabel("Metric")
                    ax[axis].set_ylabel(f"Threshold = {threshold}")
                    ax[axis].set_title(f"{args.events[axis]}")
                    ax[axis].set_yticks(np.linspace(0,1,11))
                    plt.savefig(f"Results/{save_path}/Errors/theshold_{i}")
                plt.close()

    if(plot_ROC):
        alpha = np.array(alpha)
        beta = np.array(beta)
        fig, ax = plt.subplots(1, args.num_events, figsize=(20,7))
        for axis in range(len(ax)):
            ax[axis].plot(alpha[:, axis], beta[:, axis])
            ax[axis].scatter(alpha[:, axis], beta[:, axis])
            for i, t in enumerate(thresholds):
                if i > 5 and i < 20:
                    ax[axis].annotate(f"{t:.2e}", (alpha[i, axis] + 0.01, beta[i, axis] + 0.01))
            ax[axis].set_xlabel("Probability of False Alarm")
            ax[axis].set_ylabel("Probability of Detection")
            ax[axis].set_xlim([0,1])
            ax[axis].set_ylim([0,1])
            ax[axis].set_title(f"{args.events[axis]}")
        plt.title("Region of Convergence")
        plt.show()

    if(save_spectrograms):
        print("Saving spectrograms")  
        metric = metrics(prediction,y, best_threshold)
        prediction = metric.get_thresholded_predictions()
        heigths = [10, 20, 30, 40]
        upsampler = torch.nn.Upsample(scale_factor=pooling_time_ratio, mode='nearest')
        upsampler = upsampler.to(device)
        prediction  = prediction.permute(0, 2, 1)
        prediction = upsampler(prediction)
        y = y.permute(0, 2, 1)
        y = upsampler(y)
        X = X.permute(0, 1, 3, 2)

        for i in range(X.shape[0]):
            x = X[i].squeeze(dim=0)
            x = x.cpu().detach().numpy()
            # x = np.flip(x, axis=0)
            y_ = y[i].cpu().detach().numpy() - 0.5
            pred_ = prediction[i].cpu().detach().numpy() - 0.5
            sample_rate = 1000
            window_length_samples = int(round(sample_rate * dataset_args.stft_window_seconds))
            hop_length_samples = int(round(sample_rate * dataset_args.stft_hop_seconds))
            fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 9))
            # librosa.display.specshow(x,sr=sample_rate, n_fft=fft_length, win_length=window_length_samples,hop_length=hop_length_samples, x_axis='time', cmap = None, htk=False)
            ax.imshow(x)
            ax.plot(pred_[0]*heigths[0], linewidth=3)
            ax.plot(pred_[1]*heigths[1], linewidth=3)
            ax.plot(y_[0]*heigths[2], linewidth=3)
            ax.plot(y_[1]*heigths[3], linewidth=3)
            ax.set_ylim([0, dataset_args.max_mel_band-1])
            ax.set_title("Spectrogram with predictions and targets")
            ax.set_xlabel("Time")
            ax.set_ylabel("Mel bands")
            ax.set_yticks(np.arange(0, dataset_args.max_mel_band, 2))
            plt.legend(["P Vehicle", "P Pedestrian", "T Vehicle", "T Pedestrian"])
            plt.savefig(f"Results/{save_path}/Output plots/{dataset_type}_{weights}_{i}.png")  
            plt.close()
        print("Done Saving")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "/Sep_12_2023"                                              # path to save training and testing results
    model_weights = "student_epoch_4.pt"                                    # model to evaluate
    # Real Dataset
    # eval_dataset_path = r"D:\\Purdue\\Thesis\\eaps data\\Fall 23\\EAPS\\DATASET\\Test\\Test_dataset_v1\\Real"
    # Synthetic Dataset to determine optimal thresholds
    #  eval_dataset_path = r"H:\EAPS\DATASET\Test\Synthetic_mini_v1"
    # Unlabel dataset to check generalizability
    eval_dataset_path = r"D:\Purdue\Thesis\eaps data\Fall 23\EAPS\DATASET\Test\Test_dataset_v2"
    eval_dataset_type = 'Unlabel'

    test_model(model_weights, save_path, eval_dataset_path, eval_dataset_type, 
               save_spectrograms=True, plot_ROC=False, save_metrics=False    , best_threshold=[0.3, 0.4])
