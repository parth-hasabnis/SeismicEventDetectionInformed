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

def test_model(weights, save_path, dataset_path, dataset_type, save_spectrograms: bool, plot_ROC: bool, save_metrics:bool, best_threshold=[0.5, 0.5], min_event_length=[1,2.5]):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    TEST_PATH = dataset_path    # Set dataset path 
    args = TestArguments()            # Use standard parameters

    if(not exists("Results" + save_path + "/Output Plots")):
        makedirs("Results" + save_path + "/Output Plots")
    if(not exists("Results" + save_path + "/Errors")):
        makedirs("Results" + save_path + "/Errors")


    f = open("Results" + save_path + "/dataset_args.json")  
    dataset_kwargs = json.loads(f.read())
    f.close()
    dataset_kwargs["eval"] = True
    dataset_args = DatasetArgs(**dataset_kwargs)

    seed = 75
    torch.manual_seed(seed)
    np.random.seed(seed)

    f = open("Results" + save_path + "/crnnn_args.json")
    crnn_kwargs = json.loads(f.read())
    f.close()
    pooling_time_ratio = 4  # 2 * 2
    min_event_frames = np.round(np.array(min_event_length)/pooling_time_ratio)

    crnn = CRNN(**crnn_kwargs)
    crnn.load_state_dict(torch.load(f"Results/{save_path}/Checkpoints/{weights}" ))
    crnn.to(device)
    test_dataset = SeismicEventDataset(TEST_PATH, dataset_args, dataset_type)
    eval_loader = DataLoader(test_dataset, args.batch_size, drop_last=False)

    bases = np.linspace(-5, 5, 22)          # Thresholds for predictions
    thresholds = 1/(1 + np.exp(-bases))     # Generate exponentially distributed thresholds
    thresholds = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999]

    for batch, (O, X,y) in enumerate(eval_loader):
        X = X.to(device)
        y = y.to(device)
        O = O.to(device)
        prediction, _ = crnn(X)
        if batch == 0:
            origConsolidated = O
            inputConsolidated = X
            targetConsolidated = y
            predictionConsolidated = prediction
        else:
            origConsolidated = torch.vstack((origConsolidated, O))
            inputConsolidated = torch.vstack((inputConsolidated, X))
            targetConsolidated = torch.vstack((targetConsolidated, y))
            predictionConsolidated = torch.vstack((predictionConsolidated, prediction))
    alpha = []
    beta = []
    # print(inputConsolidated.shape, targetConsolidated.shape, predictionConsolidated.shape)

    if(plot_ROC or save_metrics):
        for i, threshold in enumerate(thresholds):
            threshold = np.ones(args.num_events)*threshold
            metric = metrics(predictionConsolidated,targetConsolidated, threshold) 
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
                    plt.savefig(f"Results/{save_path}/Errors/{model_weights}_threshold_{threshold}.png")
                plt.close()

    if(plot_ROC):
        alpha = np.array(alpha)
        beta = np.array(beta)
        fig, ax = plt.subplots(1, args.num_events, figsize=(20,7))
        for axis in range(len(ax)):
            ax[axis].plot(alpha[:, axis], beta[:, axis])
            ax[axis].scatter(alpha[:, axis], beta[:, axis])
            for i, t in enumerate(thresholds):
                ax[axis].annotate(f"{t:.2f}", (alpha[i, axis] + 0.01, beta[i, axis] + 0.01))
            ax[axis].set_xlabel("Probability of False Alarm")
            ax[axis].set_ylabel("Probability of Detection")
            ax[axis].set_xlim([0,1])
            ax[axis].set_ylim([0,1])
            ax[axis].set_title(f"{args.events[axis]}")
        fig.suptitle("Receiver Operating Characteristic")
        plt.savefig(f"Results/{save_path}/Errors/ROC_{model_weights}.png")

    if(save_spectrograms):
        print("Saving spectrograms")  
        metric = metrics(predictionConsolidated,targetConsolidated, best_threshold)
        predictionConsolidated = metric.get_thresholded_predictions(min_event_frames)
        heigths = [10, 20, 30, 40]
        upsampler = torch.nn.Upsample(scale_factor=pooling_time_ratio, mode='nearest')
        upsampler = upsampler.to(device)
        predictionConsolidated  = predictionConsolidated.permute(0, 2, 1)
        predictionConsolidated = upsampler(predictionConsolidated)
        targetConsolidated = targetConsolidated.permute(0, 2, 1)
        targetConsolidated = upsampler(targetConsolidated)
        inputConsolidated = inputConsolidated.permute(0, 1, 3, 2)
        origConsolidated = origConsolidated.permute(0, 1, 3, 2)

        for i in range(origConsolidated.shape[0]):
            x = origConsolidated[i].squeeze(dim=0)
            x = x.cpu().detach().numpy()
            # x = np.flip(x, axis=0)
            y_ = targetConsolidated[i].cpu().detach().numpy() - 0.5
            pred_ = predictionConsolidated[i].cpu().detach().numpy() - 0.5
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
    save_path = "/Oct_4_2023"                                              # path to save training and testing results
    f = open("Results" + save_path + "/Checkpoints/Loss_metrics.json")
    train_metrics = json.loads(f.read())
    f.close()
    model_weights = "student_epoch_19.pt"                                    # model to evaluate

    # Real Dataset
    # eval_dataset_path = r"D:\\Purdue\\Thesis\\eaps data\\Fall 23\\EAPS\\DATASET\\Test\\Test_dataset_v1\\Real"
    # Synthetic Dataset to determine optimal thresholds
    eval_dataset_path = r"D:\\Purdue\\Thesis\\eaps data\\Fall 23\\EAPS\DATASET\\Test\\Test_dataset_v3\\Real"
    eval_dataset_type = 'Synthetic'
    # Unlabel dataset to check generalizability
    eval_dataset_path = r"D:\\Purdue\\Thesis\\eaps data\\Fall 23\\EAPS\DATASET\\Test\\Test_dataset_v2\\Unlabel"
    eval_dataset_type = 'Unlabel'

    test_model(model_weights, save_path, eval_dataset_path, eval_dataset_type, 
                save_spectrograms=True, plot_ROC=False, save_metrics=False,
                best_threshold=[0.5, 0.3], min_event_length=[10,25])
