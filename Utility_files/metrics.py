import torch

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
        self.origPredictions = self.prediction
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
        return self.origPredictions