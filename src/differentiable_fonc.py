import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA


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
    def __init__(self, pca_object, original_shape: tuple, device: str = "cpu", dtype=torch.float64):
        """
        Initialize the Differentiable PCA class with the fitted PCA sklearn object.

        Args:
            pca_object: A fitted PCA object from sklearn.
            original_shape (tuple): Shape of the input tensor (N, features, H, W).
            device (str): Device to place the PCA tensors ('cpu' or 'cuda').
            dtype (torch.dtype): Data type of the tensors.
        """
        super().__init__()
        expected_features = pca_object.components_.shape[1]
        assert original_shape[1] == expected_features, (
            f"Expected feature dimension {expected_features}, but got {original_shape[1]}"
        )
        self.n_components = pca_object.n_components
        self.original_shape = original_shape

        # Register PCA components and mean as buffers and move to the specified device
        self.register_buffer('components_', torch.tensor(pca_object.components_, dtype=dtype, device=device))
        self.register_buffer('mean_', torch.tensor(pca_object.mean_, dtype=dtype, device=device))

    @torch.no_grad()
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply PCA transformation on 4D input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, features, H, W).

        Returns:
            torch.Tensor: Transformed tensor of shape (N, n_components, H, W).
        """
        assert x.dim() == 4, f"Expected 4D tensor, but got {x.dim()}D tensor."
        assert x.shape[1] == self.original_shape[1], (
            f"Expected feature dimension {self.original_shape[1]}, but got {x.shape[1]}"
        )
        assert x.dtype == self.components_.dtype, (
            f"Expected dtype {self.components_.dtype}, but got {x.dtype}"
        )

        # Reshape to (N * H * W, features)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, self.original_shape[1])

        # Center the data
        x_centered = x_reshaped - self.mean_

        # Apply PCA transformation
        x_transformed = torch.matmul(x_centered, self.components_.T)  # Shape: (N * H * W, n_components)

        # Reshape back to (N, n_components, H, W)
        x_transformed = x_transformed.reshape(x.shape[0], x.shape[2], x.shape[3], self.n_components).permute(0, 3, 1, 2)

        return x_transformed

    @torch.no_grad()
    def inverse_transform(self, x_transformed: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the original 4D tensor from its PCA-transformed form.

        Args:
            x_transformed (torch.Tensor): Transformed tensor of shape (N, n_components, H, W).

        Returns:
            torch.Tensor: Reconstructed tensor of shape (N, features, H, W).
        """
        # Reshape to (N * H * W, n_components)
        x_transformed = x_transformed.permute(0,2,3,1).reshape(-1, self.n_components)

        # Apply inverse PCA transformation
        x_reconstructed = torch.matmul(x_transformed, self.components_) + self.mean_

        # Reshape back to original 4D shape
        x_reconstructed = x_reconstructed.reshape(self.original_shape[0], self.original_shape[2], self.original_shape[3], self.original_shape[1]).permute(0, 3, 1, 2)

        return x_reconstructed




def differentiable_min_max_search(tensor, dim = 1, tau=10):

    grad = torch.diff(tensor,dim=dim)

    grad_sign = differentiable_sign(grad,tau)

    inflection_points = torch.diff(grad_sign,dim=1)
    inflection_points = differentiable_sign(inflection_points,tau)
    inflection_points = torch.abs(inflection_points)
    inflection_points = F.pad(inflection_points,pad=(0,0,0,0,1,1),value=1)

    return inflection_points





if __name__ == "__main__":
    import numpy as np
    import torch
    from sklearn.decomposition import PCA

    # Define the mapping from string to numpy and torch data types
    dtype_mapping = {
        '32': (np.float32, torch.float32),
        '64': (np.float64, torch.float64)
    }

    # Example string value for dtype
    dtype_str = '64'

    # Get the corresponding numpy and torch data types
    np_dtype, torch_dtype = dtype_mapping[dtype_str]

    # Synthetic Data
    N, features, H, W = 2, 107, 174, 240
    x_np = np.random.rand(N, features, H, W).astype(np_dtype)  # Use the mapped numpy dtype


    # Fit PCA using sklearn
    n_components = features
    pca = PCA(n_components = n_components)  # Ensure n_components=features for exact reconstruction
    x_flat = x_np.transpose(0,2,3,1).reshape(-1, features)
    pca.fit(x_flat)

    # Initialize Differentiable4dPCA
    original_shape = x_np.shape
    device = "cpu"  # or "cuda" if available
    pca_module = Differentiable4dPCA(pca_object=pca, original_shape=original_shape, device=device, dtype=torch_dtype)

    # Convert numpy array to torch tensor
    x_tensor = torch.tensor(x_np, dtype=torch_dtype, device=device)

    # Transform
    x_transformed = pca_module.transform(x_tensor)
    print(f"Transformed shape: {x_transformed.shape}")  # Expected: (2, 107, 174, 240)

    # Compare with scikit-learn's transform
    x_transformed_sklearn = pca.transform(x_flat)
    x_transformed_sklearn_reshaped = x_transformed_sklearn.reshape(N, H, W, pca.n_components).transpose(0, 3, 1, 2)
    transformation_error = torch.mean((x_transformed - torch.tensor(x_transformed_sklearn_reshaped, dtype=torch_dtype)) ** 2).item()
    print(f"Transformation MSE: {transformation_error}")
    assert torch.allclose(x_transformed, torch.tensor(x_transformed_sklearn.reshape(N, H, W, pca.n_components).transpose(0, 3, 1, 2), dtype=torch_dtype)), "Transformation mismatch"

    # Inverse Transform
    x_reconstructed = pca_module.inverse_transform(x_transformed)
    print(f"Reconstructed shape: {x_reconstructed.shape}")  # Expected: (2, 107, 174, 240)

    # Compare with scikit-learn's inverse_transform
    x_reconstructed_sklearn = pca.inverse_transform(x_transformed_sklearn).reshape(N, H, W, features).transpose(0, 3, 1, 2)
    reconstruction_error = torch.mean((x_tensor - torch.tensor(x_reconstructed_sklearn, dtype=torch_dtype)) ** 2).item()
    print(f"Reconstruction MSE: {reconstruction_error}")
    assert torch.allclose(x_reconstructed, torch.tensor(x_reconstructed_sklearn, dtype=torch_dtype)), "Reconstruction mismatch"

