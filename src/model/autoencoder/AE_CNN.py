import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

class AE_CNN(nn.Module):
    """
    Convolutional Autoencoder class.

    Args:
        channels_list (list): List of channels for each layer.
        kernel_list (int | list): Kernel size or list of kernel sizes for each layer.
        n_conv_per_layer (int): Number of convolutional layers per block.
        padding (dict): Padding configuration.
        act_fn_str (str): Activation function name.
        final_upsample_str (str): Final upsample method.
        upsample_mode (str): Upsample mode.
        pooling (bool): Pooling method.
        pooling_dim (str): Pooling dimension.
        linear_layer (dict): Linear layer configuration.
        dropout_proba (bool): Dropout probability.
        dtype_str (str): Data type.
    """
    def __init__(self,
                 input_shape: tuple = (4, 107, 240, 240),
                 channels_list: list = [1, 1, 1, 1],
                 kernel_list: Union[int, list] = 3,
                 n_conv_per_layer: int = 1,
                 padding: dict = {"mode": "reflect", "interp_size": 20},
                 act_fn_str: str = "Relu",
                 use_final_act_fn: bool = True,
                 final_upsample_str: str = "upsample",
                 upsample_mode: str = "trilinear",
                 pooling: bool = "Max",
                 pooling_dim: str = "all",
                 linear_layer: dict = {"use": True, "cr": 1000},
                 dropout_proba: bool = 0,
                 dtype_str: str = "float32",
                 device = "cuda"):
        super().__init__()

        self.input_shape = input_shape
        self.channels_list = channels_list
        self.kernel_list = kernel_list
        self.n_conv_per_layer = n_conv_per_layer
        self.padding = padding
        self.act_fn_str = act_fn_str
        self.use_final_act_fn = use_final_act_fn
        self.final_upsample_str = final_upsample_str
        self.upsample_mode = upsample_mode
        self.pooling = pooling
        self.pooling_dim = pooling_dim
        self.linear_layer = linear_layer
        self.dropout_proba = dropout_proba
        self.dtype_str = dtype_str
        self.device = device

        self.linear_layer["latent_size"] = max(np.prod(input_shape[1:])//linear_layer["cr"],1)

        self.z_pre_linear_shape = []
        self.bottleneck_shape = []


        if isinstance(self.kernel_list, int):
            self.kernel_list = [self.kernel_list] * len(channels_list)
        elif len(self.kernel_list) < len(channels_list):
            self.kernel_list = self.kernel_list + [self.kernel_list[-1]] * (len(channels_list) - len(self.kernel_list))

        if pooling_dim == "all":
            self.pool_str = (2,2,2)
        elif pooling_dim == "depth":
            self.pool_str = (2, 1, 1)
        elif pooling_dim == "spatial":
            self.pool_str = (1, 2, 2)
        elif pooling_dim == "None":
            self.pool_str = 1
            self.channels_list = [1] * len(channels_list)


        pool_str_tensor = torch.tensor(list(self.pool_str)) ** (len(channels_list) - 1)
        self.z_pre_linear_shape = torch.Size([self.input_shape[0], self.channels_list[-1]] + torch.ceil(torch.tensor(self.input_shape[1:]) / pool_str_tensor).int().tolist())

        pooling_dict = {
            "Avg": nn.AvgPool3d(kernel_size=1, stride=self.pool_str, padding=0),
            "Max": nn.MaxPool3d(kernel_size=1, stride=self.pool_str, padding=0),
            "conv": nn.Identity(),
            "None": nn.Identity()
        }

        upsample_dict = {
            "Avg": nn.Upsample(scale_factor=self.pool_str, mode=upsample_mode),
            "Max": nn.Upsample(scale_factor=self.pool_str, mode=upsample_mode),
            "conv": nn.Identity(),
            "None": nn.Identity()
        }

        act_fn_dict = {
            "Sigmoid": nn.Sigmoid(),
            "Tanh": nn.Tanh(),
            "Softmax": nn.Softmax(dim=1),
            "Relu": nn.ReLU(),
            "Elu": nn.ELU(),
            "Gelu": nn.GELU(),
            "None": nn.Identity()
        }

        self.model_dtype = getattr(torch, dtype_str)
        self.padding_mode = padding["mode"] if (padding["mode"] in ["linear", "cubic", "reflect", "replicate", "circular"]) and (padding["interp_size"]>0) else None
        self.interp_size = padding["interp_size"] if self.padding_mode else 0


        self.pooling = pooling
        self.pooling_layer = pooling_dict[pooling]
        self.upsample_layer = upsample_dict[pooling]
        self.act_fn = act_fn_dict[act_fn_str]

        self.encoder = AE_CNN_Encoder(self)
        self.decoder = AE_CNN_Decoder(self)


        self.to(self.model_dtype)
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Reconstructed tensor.
        """
        if self.padding_mode == "linear":
            x = self.linear_padding(x, interp_size=self.interp_size)
        elif self.padding_mode == "cubic":
            x = self.cubic_padding_multi_dim(x, interp_size=self.interp_size, dim=1)
        elif self.padding_mode in ["reflect", "replicate", "circular"]:
            x = torch.nn.functional.pad(x, pad=(0, 0, 0, 0, self.interp_size, self.interp_size), mode=self.padding_mode)

        x = x.unsqueeze(1).repeat(1, self.channels_list[0], 1, 1, 1)
        z = self.encoder(x)
        self.bottleneck_shape = z.shape
        self.cr = x.shape.numel() / z.shape.numel()

        if self.linear_layer["use"]:
            self.z_pre_linear_shape = self.encoder.net(x).shape
        

        x_hat = self.decoder(z)
        x_hat = x_hat.squeeze(1)

        if self.padding_mode is not None:
            x_hat = x_hat[:, self.interp_size:-self.interp_size]

        
        return x_hat
    

    def linear_padding(self, x, interp_size=20):
        """
        Apply linear padding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            interp_size (int): Interpolation size.

        Returns:
            torch.Tensor: Padded tensor.
        """
        first_2_points = x[:, :2, :, :]
        last_2_points = x[:, -2:, :, :]
        start_pad = torch.zeros(x.size(0), interp_size, x.size(2), x.size(3), device=x.device)
        end_pad = torch.zeros(x.size(0), interp_size, x.size(2), x.size(3), device=x.device)

        for i in range(interp_size):
            start_pad[:, -(i + 1), :, :] = first_2_points[:, 0, :, :] + (i + 1) * (first_2_points[:, 0, :, :] - first_2_points[:, 1, :, :])
            end_pad[:, i, :, :] = last_2_points[:, -1, :, :] + (i + 1) * (last_2_points[:, -1, :, :] - last_2_points[:, -2, :, :])

        padded_x = torch.cat([start_pad, x, end_pad], dim=1)
        return padded_x

    def cubic_padding_multi_dim(self, signal, interp_size, dim=1):
        """
        Apply cubic padding to the input tensor along multiple dimensions.

        Args:
            signal (torch.Tensor): Input tensor.
            interp_size (int): Interpolation size.
            dim (int): Dimension along which to apply padding.

        Returns:
            torch.Tensor: Padded tensor.
        """
        signal = signal.transpose(dim, 0)
        original_shape = signal.shape
        L = original_shape[0]
        other_dims = original_shape[1:]
        N = other_dims.numel()

        signal = signal.reshape(L, N)
        x = torch.arange(L, device=signal.device, dtype=signal.dtype)
        x_start = x[:interp_size]
        y_start = signal[:interp_size, :]

        X_start = torch.stack([x_start**3, x_start**2, x_start, torch.ones_like(x_start)], dim=1)
        coeffs_start = torch.linalg.lstsq(X_start, y_start).solution

        x_pad_start = x_start - interp_size
        X_pad_start = torch.stack([x_pad_start**3, x_pad_start**2, x_pad_start, torch.ones_like(x_pad_start)], dim=1)
        y_pad_start = X_pad_start @ coeffs_start

        x_end = x[-interp_size:]
        y_end = signal[-interp_size:, :]

        X_end = torch.stack([x_end**3, x_end**2, x_end, torch.ones_like(x_end)], dim=1)
        coeffs_end = torch.linalg.lstsq(X_end, y_end).solution

        x_pad_end = x_end + interp_size
        X_pad_end = torch.stack([x_pad_end**3, x_pad_end**2, x_pad_end, torch.ones_like(x_pad_end)], dim=1)
        y_pad_end = X_pad_end @ coeffs_end

        padded_signal = torch.cat([y_pad_start, signal, y_pad_end], dim=0)
        padded_L = L + 2 * interp_size
        padded_signal = padded_signal.reshape(padded_L, *other_dims)

        dims = list(range(padded_signal.dim()))
        dims[0], dims[dim] = dims[dim], dims[0]
        padded_signal = padded_signal.permute(*dims)

        return padded_signal


class AE_CNN_Encoder(nn.Module):
    """
    Convolutional Encoder class.

    Args:
        parent (AE_CNN): Parent autoencoder class.
    """
    def __init__(self, parent):
        super().__init__()

        if parent.pooling == "conv":
            conv_stride = parent.pool_str
        else:
            conv_stride = 1

        layers = []
        for i in range(len(parent.channels_list)-1):
            ker = parent.kernel_list[i]
            pad = (ker - 1) // 2

            if parent.pooling_dim == "all":
                parent.pooling_layer.kernel_size = ker
                parent.pooling_layer.padding = pad
            elif parent.pooling_dim == "depth":
                ker = (ker, 1, 1)
                pad = (pad, 0, 0)
                parent.pooling_layer.kernel_size = ker
                parent.pooling_layer.padding = pad
            elif parent.pooling_dim == "spatial":
                ker = (1, ker, ker)
                pad = (0, pad, pad)
                parent.pooling_layer.kernel_size = ker
                parent.pooling_layer.padding = pad

            conv_layer = nn.Conv3d(in_channels=parent.channels_list[i], out_channels=parent.channels_list[i + 1], kernel_size=ker, stride=conv_stride, padding=pad, dtype=parent.model_dtype)
            layers.extend([conv_layer, parent.pooling_layer, parent.act_fn])

            if parent.n_conv_per_layer > 1:
                conv_layer = nn.Conv3d(in_channels=parent.channels_list[i + 1], out_channels=parent.channels_list[i + 1], kernel_size=ker, stride=1, padding=pad, dtype=parent.model_dtype)
                layers.extend([conv_layer, parent.act_fn] * (parent.n_conv_per_layer - 1))

        if parent.linear_layer["use"]:
            layers.extend([nn.Dropout(p=parent.dropout_proba, inplace=False)])

            latent_size = parent.linear_layer['latent_size']
            if parent.pooling_dim == "depth":
                depth_size = parent.z_pre_linear_shape[2]
                n_channels = parent.z_pre_linear_shape[1]
                layers.append(nn.Linear(n_channels * depth_size, latent_size, device=parent.device))
            elif parent.pooling_dim == "spatial":
                spatial_size = parent.z_pre_linear_shape[-2:]
                n_channels = parent.z_pre_linear_shape[1]
                layers.append(nn.Linear(n_channels * spatial_size.numel(), latent_size, device=parent.device))
            elif parent.pooling_dim == "all":
                layers.append(nn.Flatten())
                layers.append(nn.Linear(np.prod(parent.z_pre_linear_shape[1:]), latent_size, device=parent.device))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded tensor.
        """
        return self.net(x)


class AE_CNN_Decoder(nn.Module):
    """
    Convolutional Decoder class.

    Args:
        parent (AE_CNN): Parent autoencoder class.
    """
    def __init__(self, parent):
        super().__init__()

        if parent.pooling == "conv":
            conv_stride = parent.pool_str
        else:
            conv_stride = 1

        layers = []
        kernel_list = parent.kernel_list[::-1]
        channels_list = parent.channels_list[::-1]
        output_pad = 0

        if parent.linear_layer["use"]:
            latent_size = parent.linear_layer["latent_size"]
            if parent.pooling_dim == "depth":
                depth_size = parent.z_pre_linear_shape[2]
                n_channels = parent.z_pre_linear_shape[1]
                layers.append(nn.Linear(latent_size, n_channels * depth_size, device=parent.device))
                layers.append(nn.Unflatten(dim=1, unflattened_size=(n_channels, depth_size)))
                layers.append(nn.Permute(0, 3, 4, 1, 2))
            elif parent.pooling_dim == "spatial":
                spatial_size = parent.z_pre_linear_shape[-2:]
                n_channels = parent.z_pre_linear_shape[1]
                layers.append(nn.Linear(latent_size, n_channels * spatial_size.numel(), device=parent.device))
                layers.append(nn.Unflatten(dim=1, unflattened_size=(n_channels, *spatial_size)))
                layers.append(nn.Transpose(1, 2))
            elif parent.pooling_dim == "all":
                layers.append(nn.Linear(latent_size, np.prod(parent.z_pre_linear_shape[1:]), device=parent.device))
                layers.append(nn.Unflatten(dim=1, unflattened_size=(parent.z_pre_linear_shape[1:])))

        for i in range(len(channels_list)-1):
            ker = kernel_list[i]
            pad = (ker - 1) // 2

            if parent.pooling_dim == "all":
                parent.upsample_layer.kernel_size = ker
                parent.upsample_layer.padding = pad
            elif parent.pooling_dim == "depth":
                ker = (ker, 1, 1)
                pad = (pad, 0, 0)
                parent.upsample_layer.kernel_size = ker
                parent.upsample_layer.padding = pad
            elif parent.pooling_dim == "spatial":
                ker = (1, ker, ker)
                pad = (0, pad, pad)
                parent.upsample_layer.kernel_size = ker
                parent.upsample_layer.padding = pad

            if parent.pooling == "conv":
                output_pad = 1

            conv_transpose_layer = nn.ConvTranspose3d(in_channels=channels_list[i], out_channels=channels_list[i + 1], kernel_size=ker, stride=conv_stride, padding=pad, output_padding=output_pad, dtype=parent.model_dtype)
            layers.append(conv_transpose_layer)
            layers.extend([parent.upsample_layer, parent.act_fn])

            if i == len(channels_list)-2:
                if parent.final_upsample_str == "upsample_pooling":
                    layers[-2:-1] = (parent.upsample_layer, nn.AdaptiveMaxPool3d(output_size = parent.input_shape[1:]))
            
                elif parent.final_upsample_str == "upsample":
                    layers[-2] = nn.Upsample(size = parent.input_shape[1:], mode = parent.upsample_mode)


            if parent.n_conv_per_layer > 1:
                conv_layer = nn.Conv3d(in_channels=channels_list[i + 1], out_channels=channels_list[i + 1], kernel_size=ker, stride=1, padding=pad, dtype=parent.model_dtype)
                layers.extend([conv_layer, parent.act_fn] * (parent.n_conv_per_layer - 1))
        
        



        self.net = nn.Sequential(*layers)


    def forward(self, z):
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Encoded tensor.
            z_spatial_shape (list): Shape of the spatial dimensions.

        Returns:
            torch.Tensor: Reconstructed tensor.
        """


        return self.net(z)



import torch
import numpy as np

def test_non_deterministic(func, input_tensor, num_tests=10):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run the function multiple times and store the outputs
    outputs = []
    for _ in range(num_tests):
        output = func(input_tensor)
        outputs.append(output)

    # Compare the outputs
    for i in range(1, num_tests):
        if not torch.equal(outputs[0], outputs[i]):
            print("The function is non-deterministic.")
            return False

    print("The function is deterministic.")
    return True


def test_encoder_deterministic(encoder, input_tensor):
    for name, module in encoder.named_modules():
        if name:  # Skip the top-level module
            print(f"Testing module: {name}")
            if not test_non_deterministic(module, input_tensor):
                print(f"Module {name} is non-deterministic.")
                return False
            
            input_tensor = module(input_tensor)
    print("All modules in the encoder are deterministic.")
    return True


# def test_encoder_deterministic(encoder, input_tensor):
#     hooks = []
#     module_outputs = {}

#     def hook_fn(module, input, output):
#         module_outputs[module] = output

#     # Register hooks to capture the output of each module
#     for name, module in encoder.named_modules():
#         if name:
#             hooks.append(module.register_forward_hook(hook_fn))

#     # Perform a forward pass to capture the outputs
#     encoder(input_tensor)

#     # Remove hooks
#     for hook in hooks:
#         hook.remove()

#     # Test each captured output for non-determinism
#     for module, output in module_outputs.items():
#         print(f"Testing module: {module}")
#         if not test_non_deterministic(lambda x: module(x), input_tensor):
#             print(f"Module {module} is non-deterministic.")
#             return False

#         input_tensor = module_outputs[module]
#     print("All modules in the encoder are deterministic.")
#     return True

