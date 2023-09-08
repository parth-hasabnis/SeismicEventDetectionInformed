# SED_CRNN_2
# SED = Sound Event Detection
# CRNN = type of model
# 2 = Second attempt. First attempt in old working directory
# Date of creation - Aug 21 2023
# Refactoring the code to make it more user friendly 
# adding separate thresholds for events
# Run a train test loop on Aug 22 2023


import sys

import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, Sampler, DataLoader, Subset, SubsetRandomSampler, BatchSampler, random_split
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import json
from Utility_files.create_data import SeismicEventDataset, relabel_dataset, TwoStreamBatchSampler, MultiStreamBatchSampler
from models.CRNN import CRNN
from Utility_files.utils import weights_init, sigmoid_rampup, linear_rampup, Arguments, DatasetArgs
import librosa
import librosa.display

# Model was created and trained in Windows, hence command line was now used. All training parameters are set using this class

def adjust_learning_rate_2(optimizer, rampup_value, args):
    """ adjust the learning rate
    Args:
        optimizer: torch.Module, the optimizer to be updated
        rampup_value: float, the float value between 0 and 1 that should increases linearly
        rampdown_value: float, the float between 1 and 0 that should decrease linearly
    Returns:

    """
    lr = rampup_value * args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, batch_num, batches_in_epoch, args):
    lr = args.lr
    epoch = epoch + batch_num / batches_in_epoch

    lr = linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_ema_variables(model, ema_model, alpha, global_step):
    """ Update the weights of the teacher model 
    Args: 
        model: nn.Module, the student model
        ema_model: nn.Module, the teacher model
        alpha: float, the weight of the exponential
        global_step: int, the total number of minibatches trained globally
    """
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_params, params in zip(ema_model.parameters(), model.parameters()):
        ema_params.data.mul_(alpha).add_(1 - alpha, params.data)

def train(train_loader, model, optimizer, c_epoch, ema_model=None, mask_weak=None, mask_strong=None, rampup=None):
    """ One epoch of a Mean Teacher model
    Args:
        train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
            Should return a tuple: ((teacher input, student input), labels)
        model: torch.Module, model to be trained, should return a weak and strong prediction
        optimizer: torch.Module, optimizer used to train the model
        c_epoch: int, the current epoch of training
        ema_model: torch.Module, student model, should return a weak and strong prediction
        mask_weak: slice or list, mask the batch to get only the weak labeled data (used to calculate the loss)
        mask_strong: slice or list, mask the batch to get only the strong labeled data (used to calcultate the loss)
        rampup: bool, Whether or not to adjust the learning rate during training (params in config)
    """
    class_criterion = nn.BCELoss()
    consistency_criterion = nn.MSELoss()

    for i, (input, target) in enumerate(train_loader):
        global_step = c_epoch * len(train_loader) + i # total number of batches trained
        if rampup is not None:
            # rampup_value = sigmoid_rampup(global_step, args.lr_rampup*len(train_loader))
            adjust_learning_rate(optimizer, c_epoch, i, len(train_loader), args)

        if(i==0):
            print("Training!")

        batch_input = input
        noise = 0.05*torch.rand(input.shape)
        ema_batch_input = noise + input

        ema_batch_input = ema_batch_input.to(device)
        batch_input = batch_input.to(device)
        target = target.to(device)

        """ strong_pred_ema, weak_pred_ema = ema_model(ema_batch_input)
        strong_pred_ema = strong_pred_ema.detach()
        weak_pred_ema = weak_pred_ema.detach()
        strong_pred, weak_pred = model(batch_input) """

        strong_pred_ema = ema_model(ema_batch_input)
        strong_pred_ema = strong_pred_ema.detach()
        strong_pred = model(batch_input)

        if(i==0):
            print("First pass done")

        loss = None
        strong_class_loss = class_criterion(strong_pred[:args.batch_sizes[1]], target[:args.batch_sizes[1]])
        strong_ema_class_loss = class_criterion(strong_pred_ema[:args.batch_sizes[1]], target[:args.batch_sizes[1]])
        if loss is not None:
            loss += strong_class_loss
        else:
            loss = strong_class_loss

        if ema_model is not None:
            # consistency_weight = args.consistency*rampup_value
            consistency_weight = args.consistency * sigmoid_rampup(c_epoch, args.consistency_rampup)
            consistency_loss_strong = consistency_weight * consistency_criterion(strong_pred, strong_pred_ema)
            if loss is not None:
                loss += consistency_loss_strong
            else:
                loss = consistency_loss_strong

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)

    print("Training Completed")
    return loss
class metrics():

    def __init__(self, prediction:torch.Tensor, target:torch.Tensor, thresholds) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_events = target.shape[-1]
        assert all(val>0 and val<1 for val in thresholds)
        ones = torch.ones(target.shape)
        ones = ones.to(device)
        for i in range(self.n_events):
            ones[:, i] = ones[:, i] * thresholds[i]
        ones = prediction > ones
        self.prediction = ones.float()*1
        self.getPredictions = self.prediction
        self.target = target.permute(2, 0, 1)
        self.prediction = self.prediction.permute(2, 0, 1)
        
    def accuracy(self):
        return torch.sum(torch.eq(self.prediction, self.target))/self.target.nelement()
    
    def Errors(self):
        """
        Calculate the Different Errors and metrics for the training
        """

        metrics = {
            "type 1":[],
            "type 2":[],
            "precsion":[],
            "recall": []
        }

        for i in range(self.n_events):
            prediction = self.prediction[i].flatten()
            target = self.target[i].flatten()
            FALSE_ALARM_COUNT = 0
            MISS_COUNT = 0
            TRUE_POSITIVE_COUNT = 0
            TRUE_NEGATIVE_COUNT = 0

            for (p,t) in zip(prediction, target):
                if t == 1 and p == 1:
                    TRUE_POSITIVE_COUNT = TRUE_POSITIVE_COUNT + 1
                if t == 0 and p == 0:
                    TRUE_NEGATIVE_COUNT = TRUE_NEGATIVE_COUNT + 1
                if t == 1 and p == 0:
                    MISS_COUNT = MISS_COUNT + 1
                if t == 0 and p == 1:
                    FALSE_ALARM_COUNT = FALSE_ALARM_COUNT + 1
            
                TYPE_1 = FALSE_ALARM_COUNT/(FALSE_ALARM_COUNT + TRUE_NEGATIVE_COUNT + 0.00001)
                TYPE_2 = MISS_COUNT/(MISS_COUNT + TRUE_POSITIVE_COUNT + 0.00001)
                PRECISION = TRUE_POSITIVE_COUNT/(TRUE_POSITIVE_COUNT + FALSE_ALARM_COUNT + 0.00001)
                RECALL = TRUE_POSITIVE_COUNT/(TRUE_POSITIVE_COUNT + MISS_COUNT + 0.00001)     
                ACCURACY = TRUE_POSITIVE_COUNT + TRUE_NEGATIVE_COUNT/(TRUE_POSITIVE_COUNT + TRUE_NEGATIVE_COUNT + FALSE_ALARM_COUNT + MISS_COUNT)
            metrics['type 1'].append(TYPE_1)
            metrics['type 2'].append(TYPE_2)
            metrics['precsion'].append(PRECISION)
            metrics['recall'].append(RECALL)

        return metrics
    
    def get_thresholded_predictions(self):
        """
        Return the thresholded predictions 
        """
        return self.getPredictions

def test_loop(model_weights, save_spectrogtams: False, plot_ROC: True, save_metrics:False, path="First train"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_PATH = "D:\\Purdue\\Thesis\\eaps data\\2021_Urban_vibration_yamnet_V0\\Parth\\Test_data"
    args = Arguments(lr=0.0001, momentum=0.7, weight_decay=0, nesterov=False, epochs=10, exclude_unlabeled=True, 
                     consistency=0, batch_size=512, labeled_batch_size=384, batch_sizes=[384, 128])

    seed = 75
    torch.manual_seed(seed)
    np.random.seed(seed)

    f = open("Results" + path + "/crnnn_args.json")
    crnn_kwargs = json.loads(f.read())
    f.close()
    pooling_time_ratio = 4  # 2 * 2

    crnn = CRNN(**crnn_kwargs)
    crnn.load_state_dict(torch.load(f"Results/{path}/Checkpoints/{model_weights}" ))
    crnn.to(device)
    test_dataset = SeismicEventDataset(TEST_PATH, DatasetArgs, 'Synthetic')
    eval_loader = DataLoader(test_dataset, args.batch_size, drop_last=False)
    data, label = next(iter(eval_loader))

    for batch, (X, y) in enumerate(eval_loader):
        X = X.to(device)
        y = y.to(device)
        prediction = crnn(X)
    alpha = []
    beta = []
    bases = np.linspace(-5, 5, 22)
    thresholds = 1/(1 + np.exp(-bases))
        # thresholds = [0.5, 0.6, 0.7, 0.8]

    if(save_metrics):
        for i, threshold in enumerate(thresholds):
            threshold = np.ones(args.num_events)*threshold
            metric = metrics(prediction,y, threshold) 
            errors = metric.Errors()
            alpha.append(np.array(errors["type 1"]))
            beta.append(1 - np.array(errors["type 2"]))
            error = metric.Errors()
            error_values = np.array(list(error.values()))
            fig, ax = plt.subplots(nrows=1, ncols=args.num_events, figsize=(15,6))
            for axis in range(len(ax)):
                ax[axis].bar(list(error.keys()),error_values[:,axis])
                ax[axis].set_xlabel("Metric")
                ax[axis].set_ylabel(f"Threshold = {threshold}")
                ax[axis].set_title(f"{args.events[axis]}")
                ax[axis].set_yticks(np.linspace(0,1,11))
                plt.savefig(f"Results/{path}/Errors/theshold_{i}")
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

    if(save_spectrogtams):
        best_threshold = [0.85, 0.85]   
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
            window_length_samples = int(round(sample_rate * args.stft_window_seconds))
            hop_length_samples = int(round(sample_rate * args.stft_hop_seconds))
            fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 9))
            # librosa.display.specshow(x,sr=sample_rate, n_fft=fft_length, win_length=window_length_samples,hop_length=hop_length_samples, x_axis='time', cmap = None, htk=False)
            ax.imshow(x)
            ax.plot(pred_[0]*heigths[0], linewidth=3)
            ax.plot(pred_[1]*heigths[1], linewidth=3)
            ax.plot(y_[0]*heigths[2], linewidth=3)
            ax.plot(y_[1]*heigths[3], linewidth=3)
            ax.set_ylim([0, args.max_mel_band-1])
            ax.set_title("Spectrogram with predictions and targets")
            ax.set_xlabel("Time")
            ax.set_ylabel("Mel bands")
            ax.set_yticks(np.arange(0, args.max_mel_band, 2))
            plt.legend(["P Vehicle", "P Pedestrian", "T Vehicle", "T Pedestrian"])
            plt.savefig(f"Results/{path}/Output plots/Plot_{i}.png")  
            plt.close()

if __name__ == "__main__":
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       # Set Device as GPU for faster training
    save_path = "/Aug_21_2023"                                                  # path to save training and testing results
    model_weights = "student_epoch_9.pt"

    # test_loop(model_weights=model_weights, save_spectrogtams=True, plot_ROC=False,save_metrics=False, path=save_path)
    # sys.exit() 

    # Synthetic Dataset Path
    SYNTH_PATH = "D:\\Purdue\\Thesis\\eaps data\\2021_Urban_vibration_yamnet_V0\\Parth\\Strong_Dataset\\"
    # Unlabelled Dataset Path
    UNLABEL_PATH = "D:\\Purdue\\Thesis\\eaps data\\2021_Urban_vibration_yamnet_V0\\Parth\\Unlabel_Dataset\\"
    # Training arguments
    args = Arguments(lr=0.0001, momentum=0.7, weight_decay=0, nesterov=False, epochs=10, exclude_unlabeled=True, 
                     consistency=0, batch_size=64, labeled_batch_size=48, batch_sizes=[48, 16])

    seed = 75
    torch.manual_seed(seed)
    np.random.seed(seed)

    cnn_integration = False
    n_channel = 1

    #########
    # DATA
    #########

    synth_dataset = SeismicEventDataset(SYNTH_PATH, args, 'Synthetic')                                              # Load synthetic Dataset
    print("Labelled Dataset: %d", len(synth_dataset))

    synth_len = round(len(synth_dataset)*0.9)       # total samples in synthetic training dataset
    synth_dataset, valid_dataset = random_split(synth_dataset, [synth_len, len(synth_dataset) - synth_len])         # Split the synthesic dataset to create a validation dataset
    unlabel_dataset = SeismicEventDataset(UNLABEL_PATH, args, 'Unlabel')                                            # Load Unlabelled dataset
    print("UnLabelled Dataset: %d", len(unlabel_dataset))
    train_dataset = [synth_dataset, unlabel_dataset]                                                                # Create the final training dataset
    idx = 0
    indices = []
    for dataset in train_dataset:
        temp = np.arange(idx, idx+len(dataset), 1)
        idx = idx + len(dataset)
        indices.append(temp)

    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    batch_sampler = MultiStreamBatchSampler(args.subsets, indices, args.batch_sizes, args.batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    print(f"Length of training dataset {len(train_dataset)}")
    print(f"Length of validation dataset {len(valid_dataset)}")

    ########
    # MODEL
    ########

    n_layers = 6
    crnn_kwargs = {"n_in_channel": n_channel, "nclass": args.num_events, "attention": True, "n_RNN_cell": 128,
                    "n_layers_RNN": 2,
                    "activation": "glu",
                    "dropout": 0.5,
                    "cnn_integration": cnn_integration,
                    "kernel_size": n_layers * [3], "padding": n_layers * [1], "stride": n_layers * [1],
                    "nb_filters": [16,  32,  64,  128,  128, 128],
                    "pooling": [[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1,2]]}
    outfile = open("Results" + save_path + "/crnnn_args.json", "w")
    json.dump(crnn_kwargs, outfile)
    outfile.close()

    pooling_time_ratio = 4  # 2 * 2

    crnn = CRNN(**crnn_kwargs)
    crnn.apply(weights_init)
    crnn_ema = CRNN(**crnn_kwargs)
    crnn_ema.apply(weights_init)
    crnn = crnn.to(device)
    crnn_ema = crnn_ema.to(device)

    optim_kwargs = {"lr": args.lr, "betas": (0.9, 0.999)}
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
    bce_loss = nn.BCELoss()

    # Create plots to asses training performance
    trainLossLog = np.zeros((args.epochs))
    validateLossLog = np.zeros((args.epochs))
    epochsLog = np.linspace(1, args.epochs)

    ########
    # TRAIN
    ########
    for epoch in range(args.epochs):
        print(f"Starting Epoch {epoch}")
        crnn.train()
        crnn_ema.train()

        loss_value = train(train_loader, crnn, optim, epoch, ema_model=crnn_ema, rampup=True)
        print(f"Epoch {epoch}, Training Loss = {loss_value}")
        trainLossLog[epoch] = loss_value

    ###########
    # VALIDATE
    ###########
        try:
            eval_loss = 0
            for i, (X, y) in enumerate(valid_loader):
                X = X.to(device)
                y = y.to(device)
                with torch.inference_mode():
                    pred_strong = crnn(X)
                    loss = bce_loss(pred_strong, y)
                    eval_loss += loss
            
            eval_loss = eval_loss/i
            trainLossLog[epoch] = eval_loss

            print(f"Epoch {epoch}, Training Loss = {loss_value}, Validation Loss  = {eval_loss}")
        except:
            pass



        torch.save(crnn.state_dict(), f"Results/{save_path}/Checkpoints/student_epoch_{epoch}.pt")
        torch.save(crnn_ema.state_dict(), f"Results/{save_path}/Checkpoints/teacher_epoch_{epoch}.pt")


    
    