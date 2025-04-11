import torch.nn as nn
import torch.nn.functional as F

class SumSquaredError():
    '''
    MSE Loss / 2
    '''
    def __init__(self):
        super(SumSquaredError, self).__init__()

    def __call__(self, input, target):
        return F.mse_loss(input, target, reduction='mean') * 0.5