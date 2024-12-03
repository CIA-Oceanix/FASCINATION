import torch
import torch.nn as nn
import torch.nn.functional as F



class DifferentiableSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.tensor, tau: float):

        ctx.save_for_backward(input)
        ctx.tau = tau
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        tau = ctx.tau
        
        # Compute tanh with temperature in the backward pass
        input_scaled = input * tau
        tanh_grad = F.tanh(input_scaled)
        
        return grad_output * tanh_grad, None
    
    
def differentiable_sign(input, tau: float = 10):
    return DifferentiableSign.apply(input, tau)




class Differentiable4dPCA(nn.Module):
    def __init__(self, pca_object, original_shape, device = "cpu", dtype=torch.float32):
        """
        Initialize the Differentiable PCA class with the fitted PCA sklearn object.

        Args:
            pca_object: A fitted PCA object from sklearn.
            device: should be equal to x.device
            dtype: should be equal to x.dtype

        """
        super().__init__()
        self.components_ = torch.tensor(pca_object.components_, device=device, dtype=dtype)
        self.mean_ = torch.tensor(pca_object.mean_, device=device, dtype=dtype)
        self.n_components = pca_object.n_components
        self.original_shape = original_shape

    def transform(self, x):
        """
        Apply PCA transformation on 4D input tensor.

        Args:
            x: Input tensor of shape (N, 107, 174, 240).

        Returns:
            Transformed tensor of shape (N, n_components, 174, 240).
        """
        # Shape of the input: (N, 107, 174, 240)
        # Reshape to (N * 174 * 240, 107) for PCA transformation

        if self.components_.device != x.device:
            self.components_.device = x.device
            self.mean_.device = x.device

        if self.components_.dtype != x.dtype:
            self.components_ = self.components_.type(x.dtype)
            self.mean_ = self.mean_.type(x.dtype)


        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, self.original_shape[1])  # Shape (N * 174 * 240, 107)

        # Center the input data by subtracting the mean
        x_centered = x_reshaped - self.mean_

        # Perform the linear transformation using PCA components
        x_transformed = torch.matmul(x_centered, self.components_.T)  # Shape (N * 174 * 240, n_components)

        # Reshape back to (N, n_components, 174, 240)
        return x_transformed.reshape(self.original_shape[0], self.n_components, self.original_shape[2], self.original_shape[3])



    def inverse_transform(self, x_transformed):
        """
        Apply inverse PCA transformation on 4D input tensor.

        Args:
            x_transformed: Transformed tensor of shape (N, n_components, 174, 240).

        Returns:
            Inverse transformed tensor of shape (N, 107, 174, 240).
        """


        if self.components_.device != x_transformed.device:
            self.components_.device = x_transformed.device
            self.mean_.device = x_transformed.device

        if self.components_.dtype != x_transformed.dtype:
            self.components_.dtype = x_transformed.dtype
            self.mean_.dtype = x_transformed.dtype

        # Shape of the input: (N, n_components, 174, 240)


        x_transformed = x_transformed.permute(0,2,3,1).reshape(-1, self.n_components)
        # Perform the inverse transformation
        x_reconstructed = torch.matmul(x_transformed, self.components_) + self.mean_  # Shape (N, 107, 174, 240)

        return x_reconstructed.reshape(self.original_shape[0], self.original_shape[2], self.original_shape[3], self.original_shape[1]).permute(0,3,1,2)




def differentiable_min_max_search(tensor, dim = 1, tau=10):

    grad = torch.diff(tensor,dim=dim)

    grad_sign = differentiable_sign(grad,tau)

    inflection_points = torch.diff(grad_sign,dim=1)
    inflection_points = differentiable_sign(inflection_points,tau)
    inflection_points = torch.abs(inflection_points)
    inflection_points = F.pad(inflection_points,pad=(0,0,0,0,1,1),value=1)

    return inflection_points
