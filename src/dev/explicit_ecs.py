import torch 
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import src.differentiable_fonc as DF
    
    
class ECS_explicit_pred_1D(nn.Module):
    
    def __init__(self,
                 depth_array: np.array):
        super().__init__()
        self.model_dtype = getattr(torch, "float32")
        self.bias = torch.nn.Parameter(torch.empty(0))
        self.depth_array = depth_array
        self.tau = 100
    
    def forward(self, ssp):
        # ssp shape: [batch_size*240*240, 1, 107]
        ssp = ssp.unsqueeze(1)  # This operation might be unnecessary if ssp already has the correct shape
        kernel = torch.tensor([-1.0, 1.0]).float().view(1, 1, 2).to(ssp.device)
        derivative = F.conv1d(ssp, kernel, padding=0)

        sign = torch.sign(derivative) + F.tanh(self.tau * derivative) - F.tanh(self.tau * derivative).detach()

        sign_diff = F.conv1d(sign, kernel, padding=1)
        sign_change = F.tanh(10 * F.relu(-sign_diff))

        for pattern in ([1, 0, 1], [1, -1, 0, 0]):
            n = len(pattern)
            kernel_matrix = torch.eye(n)
            element_match = 0
            for i in range(n):
                kernel_element = kernel_matrix[i, :].view(1, 1, n).to(ssp.device)
                element_match += (F.conv1d(sign, kernel_element, padding=0) - pattern[i]) ** 2

            # Adjust padding to match the length of sign_change
            pattern_recognition = F.pad(element_match, (1, sign_change.shape[2] - element_match.shape[2] - 1), value=1.0)
            mask_discontinuity = 1 - F.relu(pattern_recognition + 1) * F.relu(1 - pattern_recognition)

            sign_change = sign_change * mask_discontinuity

        mask = F.relu(2 - torch.cumsum(sign_change, dim=2))

        # Expand and align depth_array with the reduced shape of the input tensor
        depth_array_tens = torch.tensor(np.expand_dims(self.depth_array[:mask.shape[2]], axis=0)).to(ssp.device).type(sign_change.dtype)
        depth_array_tens[0, 0] = 0.0  # TODO: Handle the first depth value properly

        ecs_pred = (sign_change * mask).squeeze(dim=1)
        ecs_pred = (ecs_pred * depth_array_tens).max(dim=1).values / 670.25141631

        return ecs_pred
    
    
    
    
def explicit_ecs_2D(ssp: torch.tensor,
                    depth_tens: torch.tensor,
                    batch: bool = False,
                    tau = 100):
    
    if batch:
        ssp = ssp.unsqueeze(1).nan_to_num()  
    
    else:
        ssp = ssp.unsqueeze(0).unsqueeze(0).nan_to_num()  
        
    kernel = torch.tensor([-1.0, 1.0]).float().view(1, 1, 2, 1).to(ssp.device)
    derivative = F.conv2d(ssp, kernel, padding=0)

    sign = torch.sign(derivative) + F.tanh(tau * derivative) - F.tanh(tau * derivative).detach()

    sign_diff = F.conv2d(sign, kernel, padding=(1,0))
    sign_change = F.tanh(10 * F.relu(-sign_diff))

    for pattern in ([1, 0, 1], [1, -1, 0, 0]):
        n = len(pattern)
        kernel_matrix = torch.eye(n)
        element_match = 0
        for i in range(n):
            kernel_element = kernel_matrix[i, :].view(1, 1, n, 1).to(ssp.device)
            element_match += (F.conv2d(sign, kernel_element, padding=0) - pattern[i]) ** 2

        # Adjust padding to match the length of sign_change
        pattern_recognition = F.pad(element_match, (0,0,1, sign_change.shape[2] - element_match.shape[2] - 1), value=1.0)
        mask_discontinuity = 1 - F.relu(pattern_recognition + 1) * F.relu(1 - pattern_recognition)

        sign_change = sign_change * mask_discontinuity

    mask = F.relu(2 - torch.cumsum(sign_change, dim=2))

    # Expand and align depth_array with the reduced shape of the input tensor
     
    depth_array_tens = depth_tens[:mask.shape[2]].view(1,-1, 1).to(ssp.device).type(sign_change.dtype)
    depth_array_tens[0, 0] = 0.0  # TODO: Handle the first depth value properly

    ecs_pred = (sign_change * mask).squeeze(dim=1)
    ecs_pred = (ecs_pred * depth_array_tens).max(dim=1).values
    
    if not batch:
        ecs_pred = ecs_pred.squeeze(0)
        
    return ecs_pred

#ecs_2d = explicit_ecs_2D(torch.tensor(ssp_truth_unorm_test_arr[t,:,lat,:]).float(),torch.tensor(depth_array).float()).to(device) 





class ECS_explicit_pred_3D(nn.Module):
    
    def __init__(self,
                 depth_array: np.array):
        super().__init__()
        self.model_dtype = getattr(torch, "float32")
        self.bias = torch.nn.Parameter(torch.empty(0))
        #self.bias.requires_grad = False
        self.depth_array = depth_array
        self.tau = 10
    
    def forward(self,ssp):
        
        ssp = ssp.unsqueeze(1)       
 
        kernel = torch.tensor([-1.0, 1.0], device=ssp.device, requires_grad=False).float().view(1,1,2,1,1)
        derivative = F.conv3d(ssp, kernel, padding=(0,0,0))
        sign = DF.differentiable_sign(derivative, self.tau)
        #sign = torch.sign(derivative) + F.tanh(self.tau * derivative) - F.tanh(self.tau * derivative).detach()
        

        sign_diff = F.conv3d(sign, kernel, padding=(1,0,0))
        sign_change = DF.differentiable_sign(F.relu(-sign_diff), self.tau)
        sign_change[:,:,0,:,:] =  0 ##TODOm: check if this is correct
        #sign_change = F.tanh(10*F.relu(-sign_diff))

        for pattern in ([1, 0, 1], [1, -1, 0, 0]):  
            n = len(pattern)
            kernel_matrix = torch.eye(n)
            element_match = 0
            for i in range(n):
                kernel_element = kernel_matrix[i,:].view(1,1,n,1,1).to(ssp.device)
                element_match = element_match + (F.conv3d(sign, kernel_element, padding=(0,0,0)) - pattern[i])**2

            pattern_recognition = F.pad(element_match, (0, 0, 0, 0, 1, (sign_change.shape[2]- element_match.shape[2]) - 1),value=1.)    
            mask_discontinuity = 1 - F.relu(pattern_recognition+1) * F.relu(1-pattern_recognition)

            sign_change = sign_change * mask_discontinuity


        mask = F.relu(2 - torch.cumsum(sign_change, dim=2))

        depth_array_tens = torch.tensor(np.expand_dims(self.depth_array[:mask.shape[2]], axis = (0,2,3))).to(ssp.device).type(sign_change.dtype)
        depth_array_tens[0,0,0,0] = 0.  ##TODO the true first z value is equal to 48cm. It may have to be considered that way
        ecs_pred = (sign_change * mask ).squeeze(dim=1)
        ecs_pred = (ecs_pred * depth_array_tens).max(dim=1).values /670.25141631

        return ecs_pred
    
    
    
