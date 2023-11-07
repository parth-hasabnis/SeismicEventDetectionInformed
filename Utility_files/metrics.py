import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import numpy as np

class metrics():

    def __init__(self, prediction:torch.Tensor, target:torch.Tensor, thresholds) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_events = target.shape[-1]
        assert all(val>0 and val<1 for val in thresholds)
        ones = torch.ones(target.shape)
        ones = ones.to(device)
        for i in range(self.n_events):
            ones[:,:, i] = ones[:,:, i] * thresholds[i]
        ones = prediction > ones
        self.prediction = ones.float()*1
        self.origPredictions = self.prediction      #  Thresholded, but before min_event_length
        self.target = target.permute(2, 0, 1)
        self.prediction = self.prediction.permute(2, 0, 1)
        
    def accuracy(self):
        return torch.sum(torch.eq(self.prediction, self.target))/self.target.nelement()
    
    def Errors(self):
        """
        Calculate the Different Errors and metrics for the training
        """

        errors = np.zeros((self.n_events, 4))
        for i in range(self.n_events):
            prediction = self.prediction[i].flatten()
            prediction = prediction.cpu()
            target = self.target[i].flatten()
            target = target.cpu()
            errors[i, :] = errors[i, :] + confusion_matrix(target, prediction).ravel()
            
        return errors
    
    def get_thresholded_predictions(self, min_event_length):
        """
        Return the thresholded predictions 
        """
        assert len(min_event_length) == self.n_events
        predictions  = self.origPredictions.permute(0, 2, 1)
        predictions = predictions.cpu().detach().numpy()
        predictions = predictions.astype(int)
        # Switch off events based on background
        for event in range(self.n_events-1):
            predictions[:,event] = predictions[:,event] & ~(predictions[:,-1])

        summ=0
        for k, pred in enumerate(predictions):
            for j, channel in enumerate(pred):
                for i, elem in enumerate(channel):
                    if elem != 0:
                        summ = summ + elem
                        if i == predictions.shape[2]-1:
                            if summ<min_event_length[j]:
                                predictions[k][j][i-summ+1:i+1] = 0
                    else:
                        if summ:
                            if summ<min_event_length[j]:
                                # print(k,j,i-summ, i)
                                predictions[k][j][i-summ:i] = 0
                        summ = 0
                summ = 0
            summ = 0

        new_predictions = torch.Tensor(predictions).float().permute(0,2,1)
        return(new_predictions)
    