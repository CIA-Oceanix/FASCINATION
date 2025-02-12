
import torch
import torch.nn as nn
import src.differentiable_fonc as DF

class DiceLoss(nn.Module):
    def __init__(self, epsilon):
        
        self.epsilon = epsilon
        super().__init__()

    def forward(self, inputs, outputs):
        
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        outputs = outputs.view(-1)
        
        intersection = (inputs * outputs).sum()                          
        dice = (2.*intersection + self.epsilon)/(inputs.sum() + outputs.sum() + self.epsilon)  
        
        return 1 - dice
    
    

class BCELoss(nn.Module):
    def __init__(self, weight = None, reduction= "mean"):
        super().__init__()
        
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, outputs):
        
        weight_tensor = torch.tensor(self.weight).to(inputs.device)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        outputs = outputs.view(-1)

        return nn.BCELoss(weight = weight_tensor, reduction=self.reduction)(inputs, outputs)


class DiceBCELoss(nn.Module):
    def __init__(self, weight = None, reduction= "mean", epsilon = 1):
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        self.epsilon = epsilon
        
        
    def forward(self, inputs, outputs):
        

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        outputs = outputs.view(-1)
        
        bce_loss = BCELoss(weight = self.weight, reduction = self.reduction)(inputs, outputs)               
        dice_loss = DiceLoss(epsilon = self.epsilon)(inputs, outputs)
        Dice_BCE = bce_loss + dice_loss
        
        return Dice_BCE
    


def fourier_loss(outputs, inputs):
    outputs_fft = torch.fft.fft(outputs, dim=2)
    inputs_fft = torch.fft.fft(inputs, dim=2)
    return torch.mean((torch.abs(outputs_fft) - torch.abs(inputs_fft)) ** 2)



def weighted_mse_loss(outputs, inputs, depth_tens, significant_depth, decay_factor = 0.1):

    signal_length = len(depth_tens)

    weights = torch.ones(signal_length, device=inputs.device, dtype=inputs.dtype)

    max_significant_depth_idx = torch.searchsorted(depth_tens, significant_depth, right=False)

    weights[:max_significant_depth_idx] = 1.0  # Strong emphasis on the first 30 points
    weights[max_significant_depth_idx+1:] = torch.exp(-decay_factor * torch.arange(max_significant_depth_idx+1, signal_length))

    # Reshape to match inputs shape
    weights = weights.view(1, 1, -1, 1, 1)  # Shape: [1, 1, signal_length, 1, 1]

    weighted_loss =  torch.mean(weights * (outputs - inputs) ** 2)
    return weighted_loss



def max_position_and_value_loss(inputs,outputs, depth_dim=1):

        inputs_max_value, inputs_max_pos = torch.max(inputs, dim=depth_dim)
        outputs_max_value, outputs_max_pos = torch.max(outputs, dim=depth_dim)

        max_position_loss =  nn.MSELoss()(inputs_max_pos.float(), outputs_max_pos.float()) 
        max_value_loss =  nn.MSELoss()(inputs_max_value, outputs_max_value) 

        return max_position_loss, max_value_loss


def min_max_position_and_value_loss(inputs,outputs, depth_dim=1, tau = 10):

    signal_length = inputs.shape[1]
    min_max_inputs_mask = DF.differentiable_min_max_search(inputs,dim=depth_dim,tau=tau)
    min_max_outputs_mask = DF.differentiable_min_max_search(outputs,dim=depth_dim,tau=tau)
    index_tensor = torch.arange(0, signal_length,device=inputs.device, dtype=inputs.dtype).view(1, -1, 1, 1)
    truth_inflex_pos = (min_max_inputs_mask * index_tensor).sum(dim=depth_dim)/min_max_inputs_mask.sum(dim=depth_dim)
    pred_inflex_pos = (min_max_outputs_mask * index_tensor).sum(dim=depth_dim)/min_max_outputs_mask.sum(dim=depth_dim)

    min_max_pos_loss = nn.MSELoss()(pred_inflex_pos, truth_inflex_pos)
    min_max_value_loss = nn.MSELoss(reduction="none")(outputs,inputs)*min_max_inputs_mask
    min_max_value_loss = min_max_value_loss.mean()

    return min_max_pos_loss, min_max_value_loss



def gradient_mse_loss(inputs, outputs, depth_tens, depth_dim=1):
    assert len(depth_tens)>1, "Depth tensor must have more than one element"
    coordinates = (depth_tens,)
    ssp_gradient_inputs = torch.gradient(input = inputs, spacing = coordinates, dim=depth_dim)[0]
    ssp_gradient_outputs = torch.gradient(input = outputs, spacing = coordinates, dim=depth_dim)[0]

    gradient_loss =  nn.MSELoss()(ssp_gradient_inputs, ssp_gradient_outputs) 
    return gradient_loss

