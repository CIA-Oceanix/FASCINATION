
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, epsilon):
        
        self.epsilon = epsilon
        super().__init__()

    def forward(self, inputs, targets):
        
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                          
        dice = (2.*intersection + self.epsilon)/(inputs.sum() + targets.sum() + self.epsilon)  
        
        return 1 - dice
    
    

class BCELoss(nn.Module):
    def __init__(self, weight = None, reduction= "mean"):
        super().__init__()
        
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        
        weight_tensor = torch.tensor(self.weight).to(inputs.device)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        return nn.BCELoss(weight = weight_tensor, reduction=self.reduction)(inputs, targets)


class DiceBCELoss(nn.Module):
    def __init__(self, weight = None, reduction= "mean", epsilon = 1):
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        self.epsilon = epsilon
        
        
    def forward(self, inputs, targets):
        

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        bce_loss = BCELoss(weight = self.weight, reduction = self.reduction)(inputs, targets)               
        dice_loss = DiceLoss(epsilon = self.epsilon)(inputs, targets)
        Dice_BCE = bce_loss + dice_loss
        
        return Dice_BCE
    


def fourier_loss(output, target):
    output_fft = torch.fft.fft(output, dim=2)
    target_fft = torch.fft.fft(target, dim=2)
    return torch.mean((torch.abs(output_fft) - torch.abs(target_fft)) ** 2)


def weighted_mse_loss(output, target, weight):
    return torch.mean(weight * (output - target) ** 2)

# Create a decay weight vector emphasizing the first 30 points
signal_length = 107
decay_factor = 0.1  # This controls the decay rate
weights = torch.ones(signal_length)
weights[:30] = 1.0  # Strong emphasis on the first 30 points
weights[30:] = torch.exp(-decay_factor * torch.arange(30, signal_length))

# Reshape to match input shape
weights = weights.view(1, 1, -1, 1, 1)  # Shape: [1, 1, signal_length, 1, 1]
