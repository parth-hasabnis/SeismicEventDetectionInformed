# File to evalute the performance of trained Models


import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import Utility_files.metrics as metrics
import Utility_files.test_args as test_args
from Utility_files.create_data import SeismicEventDataset, relabel_dataset, TwoStreamBatchSampler, MultiStreamBatchSampler
from models.CRNN import CRNN
from Utility_files.utils import weights_init, sigmoid_rampup, linear_rampup
import librosa
import librosa.display

def test_model(weights, save_path, dataset_path, dataset_type, save_spectrograms, plot_ROC, save_metrics):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_PATH = dataset_path    # Set dataset path 
    args = test_args            # Use standard parameters

    seed = 75
    torch.manual_seed(seed)
    np.random.seed(seed)

    f = open("Results" + save_path + "/crnnn_args.json")
    crnn_kwargs = json.loads(f.read())
    f.close()
    pooling_time_ratio = 4  # 2 * 2

    crnn = CRNN(**crnn_kwargs)
    crnn.load_state_dict(torch.load(f"Results/{save_path}/Checkpoints/{model_weights}" ))
    crnn.to(device)
    test_dataset = SeismicEventDataset(TEST_PATH, args, dataset_type)
    eval_loader = DataLoader(test_dataset, args.batch_size, drop_last=False)

    bases = np.linspace(-5, 5, 22)          # Thresholds for predictions
    thresholds = 1/(1 + np.exp(-bases))     # Generate exponentially distributed thresholds

    if(plot_ROC):
        yConsolidated = []
        predictionConsolidated = []    

    for batch, (X,y) in enumerate(eval_loader):

        X = X.to(device)
        y = y.to(device)
        prediction = crnn(X)






if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "/Aug_21_2023"                                              # path to save training and testing results
    model_weights = "student_epoch_9.pt"                                    # model to evaluate

