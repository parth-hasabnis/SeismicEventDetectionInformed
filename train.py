# Train any model with any parameters
# Constructed using CRNN as reference, but can be modified 
# Date of creation - Sep 08 2023

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
from Utility_files.create_data import SeismicEventDataset, relabel_dataset, TwoStreamBatchSampler, MultiStreamBatchSampler
from torch.utils.data import random_split, ConcatDataset
from models.CRNN import CRNN
from Utility_files.utils import weights_init, sigmoid_rampup, linear_rampup, Arguments, DatasetArgs
import traceback

def adjust_learning_rate(optimizer, epoch, batch_num, batches_in_epoch, args):
    lr = args.lr
    epoch = epoch + batch_num/batches_in_epoch

    if(epoch < args.lr_rampup):
        '''
        Linear Learning rate ramp-up per batch
        y = m*x + c
        y = learning rate for current epoch
        m = slope
        x = fractional epoch (epoch + current_batch/batches_in_epoch)
        c = initial learning rate
        '''
        m = (args.lr - args.initial_lr)/args.lr_rampup      
        c = args.initial_lr

        lr = m*epoch + c
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

def train_one_epoch(train_loader, model, optimizer, c_epoch, ema_model=None, mask_weak=None, mask_strong=None, rampup=None):
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

        ema_batch_input = ema_batch_input.to(device=device, dtype=torch.float32)
        batch_input = batch_input.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)

        if ema_model is not None:
            strong_pred_ema, weak_pred_ema = ema_model(ema_batch_input)
            strong_pred_ema = strong_pred_ema.detach()
            weak_pred_ema = weak_pred_ema.detach()
        strong_pred, weak_pred = model(batch_input)

        strong_pred_trim = strong_pred[:args.batch_sizes[0]]
        weak_pred_trim = weak_pred[args.batch_sizes[0]:]

        loss = None
        if mask_weak == None:
            strong_class_loss = class_criterion(strong_pred[:args.batch_sizes[0]], target[:args.batch_sizes[0]])            # Discard the outputs of unlabelled data
            if ema_model is not None:
                strong_ema_class_loss = class_criterion(strong_pred_ema[:args.batch_sizes[0]], target[:args.batch_sizes[0]])
        if loss is not None:
            loss += strong_class_loss
        else:
            loss = strong_class_loss
        if ema_model is not None:
            if mask_strong == None:
                consistency_weight = args.consistency * sigmoid_rampup(c_epoch, args.consistency_rampup)
                consistency_loss_weak = consistency_weight * consistency_criterion(weak_pred[args.batch_sizes[0]:], weak_pred_ema[args.batch_sizes[0]:])
            if loss is not None:
                loss += consistency_loss_weak
            else:
                loss = consistency_loss_weak

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)

    return loss

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    save_path = "/Sep_22_2023_2"
    
    SYNTH_PATH = r"D:\\Purdue\\Thesis\\eaps data\\Fall 23\\EAPS\DATASET\\Train\\Strong_Dataset_v3"
    UNLABEL_PATH = r"D:\\Purdue\\Thesis\\eaps data\\Fall 23\\EAPS\DATASET\\Train\\Unlabel_Dataset_v2"

    kwargs = {"lr":0.001, "momentum":0.7, "nesterov":True, "epochs":15, "exclude_unlabelled":True,
              "consistency":20, "batch_size":800, "labeled_batch_size":600,"batch_sizes":[600, 200],
               "initial_lr":0.00001, "lr_rampup":7, "consistency_rampup":15, "weight_decay":0}
    args = Arguments(**kwargs)
    outfile = open("Results" + save_path + "/training_args.json", "w")
    json.dump(kwargs, outfile, indent=2)
    outfile.close()

    dataset_kwargs = {"num_events":2, "max_mel_band":64, "stft_window_seconds":0.25, "stft_hop_seconds":0.1,
                      "mel_bands": 64, "sample_rate":500, "power":2, "normalize": False, "mel_offset": 2 
    }
    outfile = open("Results" + save_path + "/dataset_args.json", "w")
    json.dump(dataset_kwargs, outfile, indent=2)
    outfile.close()
    dataset_args = DatasetArgs(**dataset_kwargs)

    cnn_integration = False
    n_channel = 1

    #########
    # DATA
    #########

    synth_dataset = SeismicEventDataset(SYNTH_PATH, dataset_args, 'Synthetic')                                              # Load synthetic Dataset
    print("Labelled Dataset:", len(synth_dataset))

    synth_len = round(len(synth_dataset)*0.7)       # total samples in synthetic training dataset
    synth_dataset, valid_dataset = random_split(synth_dataset, [synth_len, len(synth_dataset) - synth_len])         # Split the synthesic dataset to create a validation datase

    if args.exclude_unlabelled == False:
        unlabel_dataset = SeismicEventDataset(UNLABEL_PATH, dataset_args, 'Unlabel')                                            # Load Unlabelled dataset
        print("UnLabelled Dataset:", len(unlabel_dataset))

    ### Test on small dummy data
    '''
    synth_dataset, sm = random_split(synth_dataset, [args.labeled_batch_size, len(synth_dataset) - args.labeled_batch_size]) 
    if args.exclude_unlabelled == False:
        unlabel_dataset, sm = random_split(unlabel_dataset, [args.batch_sizes[1], len(unlabel_dataset) - args.batch_sizes[1]])
    valid_dataset, sm = random_split(valid_dataset, [args.batch_sizes[1], len(valid_dataset) - args.batch_sizes[1]]) 
    '''
    ###

    if args.exclude_unlabelled == False:
        train_dataset = [synth_dataset, unlabel_dataset]  
        idx = 0
        indices = []
        for dataset in train_dataset:
            temp = np.arange(idx, idx+len(dataset), 1)
            idx = idx + len(dataset)
            indices.append(temp) 
        train_dataset = ConcatDataset(train_dataset)
    else:
        train_dataset = synth_dataset
        indices = []
        temp = np.arange(0, len(train_dataset))    
        indices.append(temp)                                                        # Create the final training dataset
        
    batch_sampler = MultiStreamBatchSampler(args.subsets, indices, args.batch_sizes, args.batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=2)
    print(f"Length of training dataset {len(train_dataset)}")
    print(f"Length of validation dataset {len(valid_dataset)}")


    ########
    # MODEL
    ########

    n_layers = 6
    crnn_kwargs = {"n_in_channel": n_channel, "nclass": dataset_args.num_events, "attention": True, "n_RNN_cell": 128,
                    "n_layers_RNN": 2,
                    "activation": "glu",
                    "dropout": 0.5,
                    "cnn_integration": cnn_integration,
                    "kernel_size": n_layers * [3], "padding": n_layers * [1], "stride": n_layers * [1],
                    "nb_filters": [16,  32,  64,  128,  128, 128],
                    "pooling": [[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1,2]]}
    outfile = open("Results" + save_path + "/crnnn_args.json", "w")
    json.dump(crnn_kwargs, outfile, indent=2)
    outfile.close()

    pooling_time_ratio = 4  # 2 * 2

    crnn = CRNN(**crnn_kwargs)
    crnn.apply(weights_init)
    if args.exclude_unlabelled == False: 
        crnn_ema = CRNN(**crnn_kwargs)
        crnn_ema.apply(weights_init)
        crnn_ema = crnn_ema.to(device)
    else:
        crnn_ema = None
    crnn = crnn.to(device)

    optim_kwargs = {"lr": args.lr, "betas": (0.9, 0.999)}
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
    bce_loss = nn.BCELoss()

    # Create plots to asses training performance
    trainLossLog = np.zeros((args.epochs))
    validateLossLog = np.zeros((args.epochs))
    lrLog = np.zeros((args.epochs))
    epochsLog = np.linspace(1, args.epochs, args.epochs)
    bestLoss = torch.tensor(float('inf'))

    ########
    # TRAIN
    ########
    try: 
        for epoch in range(args.epochs):
            print(f"Starting Epoch {epoch}")
            crnn.train()
            if args.exclude_unlabelled == False:
                crnn_ema.train()

            loss_value = train_one_epoch(train_loader, crnn, optim, epoch, ema_model=crnn_ema, rampup=True)
            print(f"Epoch {epoch}, Training Loss = {loss_value}")
            trainLossLog[epoch] = loss_value

            for param_group in optim.param_groups:
                lrVal = param_group['lr'] 
                lrLog[epoch] = lrVal

    ###########
    # VALIDATE
    ###########
            eval_loss = 0
            for i, (X, y) in enumerate(valid_loader):
                X = X.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)
                with torch.inference_mode():
                    pred_strong, pred_weak = crnn(X)
                    loss = bce_loss(pred_strong, y)
                    eval_loss += loss
            
            eval_loss = eval_loss/i
            validateLossLog[epoch] = eval_loss

            print(f"Epoch {epoch}, Training Loss = {loss_value}, Validation Loss  = {eval_loss}")

            torch.save(crnn.state_dict(), f"Results/{save_path}/Checkpoints/student_epoch_{epoch}.pt")
            if args.exclude_unlabelled == False:
                torch.save(crnn_ema.state_dict(), f"Results/{save_path}/Checkpoints/teacher_epoch_{epoch}.pt")

            if(eval_loss < bestLoss):
                torch.save(crnn.state_dict(), f"Results/{save_path}/Checkpoints/best_model.pt")
                bestLoss = eval_loss
    except Exception as e: 
        print(traceback.format_exc())
    finally:
        loss_metrics = {"train":list(trainLossLog), 
                        "valid":list(validateLossLog), 
                        "epochs":list(epochsLog),
                        "learning_rate":list(lrLog)}
        with open(f"Results/{save_path}/Checkpoints/Loss_metrics.json", "w") as outfile:
            json.dump(loss_metrics, outfile, indent=2)


    






