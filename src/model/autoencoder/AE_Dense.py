import torch
import torch.nn as nn
import torch.nn.functional as F

# Removed helper modules as they are no longer needed
# ...existing code...

# New encoder for AE_Dense
class AE_Dense_Encoder(nn.Module):
    def __init__(self, input_features, features_list, act_fn, dropout_proba):
        super().__init__()
        layers = []
        in_features = input_features
        # features_list is expected to be [input_features, ...]
        for i in range(len(features_list) - 1):
            out_features = features_list[i + 1]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(act_fn)
            if dropout_proba > 0:
                layers.append(nn.Dropout(p=dropout_proba))
            in_features = out_features
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# New decoder for AE_Dense
class AE_Dense_Decoder(nn.Module):
    def __init__(self, features_list, act_fn, dropout_proba):
        super().__init__()
        rev_features = features_list[::-1]
        layers = []
        in_features = rev_features[0]
        for i in range(len(rev_features) - 1):
            out_features = rev_features[i + 1]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(act_fn)
            if dropout_proba > 0:
                layers.append(nn.Dropout(p=dropout_proba))
            in_features = out_features
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Modified AE_Dense using the new encoder and decoder classes.
class AE_Dense(nn.Module):
    """
    Fully connected autoencoder for 1D input (features).
    
    Args:
        input_shape (tuple): Tuple with one element representing the number of features.
        features_list (list): List of feature dimensions. The provided list now defines the out_features for each layer.
        act_fn_str (str): Activation function ("LeakyRelu", "Relu", etc.).
        dropout_proba (float): Dropout probability.
        norm_stats (dict): normalization config dict. Default: None meaning normalization/unnorm will occur in AE_Dense later.
    """
    def __init__(self,
                 input_shape: tuple = (64, 300),
                 features_list: list = [32, 16],
                 act_fn_str: str = "LeakyRelu",
                 dropout_proba: float = 0.0,
                 norm_stats: dict = None):
        super().__init__()
        self.input_features = input_shape[-1]
        # Prepend input_features so that features_list becomes [input_features, ...]
        features_list[-1] = min(features_list[-1], self.input_features)
        features_list = [self.input_features] + features_list

        act_fn_dict = {
            "LeakyRelu": nn.LeakyReLU(),
            "Relu": nn.ReLU(),
            "Sigmoid": nn.Sigmoid(),
            "Tanh": nn.Tanh(),
            "None": nn.Identity()
        }
        self.act_fn = act_fn_dict.get(act_fn_str, nn.LeakyReLU())
        self.dropout_proba = dropout_proba

        self.encoder = AE_Dense_Encoder(self.input_features, features_list, self.act_fn, dropout_proba)
        self.decoder = AE_Dense_Decoder(features_list, self.act_fn, dropout_proba)

        # Store norm_stats in case later updated via autoencoder_V2.initiate_model:
        self.norm_stats = norm_stats
        # By default the normalization layers are empty (to be updated later)
        self.norm_layer = None  
        self.unnorm_layer = None  

    def forward(self, x):
        # Optionally, if norm_layer is not None, apply normalization here.
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x_enc = self.encoder(x)
        self.bottleneck_shape = x_enc.shape
        self.cr = self.input_features / x_enc.shape[-1]
        x_dec = self.decoder(x_enc)
        # Optionally, if unnorm_layer is not None, apply unnormalization here.
        if self.unnorm_layer is not None:
            x_dec = self.unnorm_layer(x_dec)
        return x_dec

    def update_norm_layers(self, norm_stats):
        """
        Updates self.norm_layer and self.unnorm_layer based on norm_stats.
        """
        self.norm_stats = norm_stats
        method = norm_stats["method"]
        if method == "min_max":
            x_min, x_max = norm_stats["params"]["x_min"], norm_stats["params"]["x_max"]
            self.norm_layer = LambdaLayer(lambda x: (x - x_min) / (x_max - x_min) if (x_max - x_min) != 0 else x)
            self.unnorm_layer = LambdaLayer(lambda x: x * (x_max - x_min) + x_min)
        elif method == "mean_std":
            mean, std = norm_stats["params"]["mean"], norm_stats["params"]["std"]
            self.norm_layer = LambdaLayer(lambda x: (x - mean) / std)
            self.unnorm_layer = LambdaLayer(lambda x: x * std + mean)
        elif method == "mean_std_along_depth":
            self.norm_layer = LambdaLayer(lambda x: (x - torch.tensor(norm_stats["params"]["mean"].reshape(1,-1),
                                                                  device=x.device, dtype=x.dtype)) / 
                                                   torch.tensor(norm_stats["params"]["std"].reshape(1,-1),
                                                                  device=x.device, dtype=x.dtype))
            self.unnorm_layer = LambdaLayer(lambda x: x * torch.tensor(norm_stats["params"]["std"].reshape(1,-1),
                                                                      device=x.device, dtype=x.dtype) +
                                                         torch.tensor(norm_stats["params"]["mean"].reshape(1,-1),
                                                                      device=x.device, dtype=x.dtype))
        else:
            self.norm_layer = nn.Identity()
            self.unnorm_layer = nn.Identity()

# Add a simple LambdaLayer wrapper for lambda functions.
class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

















# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Helper module for pooling over the profiles dimension
# class PoolProfiles(nn.Module):
#     def __init__(self, pooling: str, kernel_size: int):
#         super().__init__()
#         if pooling == "Max":
#             self.pool = nn.MaxPool1d(kernel_size)
#         elif pooling == "Avg":
#             self.pool = nn.AvgPool1d(kernel_size)
#         else:
#             self.pool = None

#     def forward(self, x):
#         # x shape: (batch, profiles, features)
#         if self.pool:
#             x = x.permute(0, 2, 1)  # (B, features, profiles)
#             x = self.pool(x)
#             x = x.permute(0, 2, 1)
#         return x

# # Helper module for upsampling over the profiles dimension
# class UpsampleProfiles(nn.Module):
#     def __init__(self, target_size: int):
#         super().__init__()
#         self.target_size = target_size

#     def forward(self, x):
#         # x shape: (batch, profiles, features)
#         x = x.permute(0, 2, 1)  # (B, features, profiles)
#         x = F.interpolate(x, size=self.target_size, mode='nearest')
#         x = x.permute(0, 2, 1)
#         return x

# class AE_Dense(nn.Module):
#     """
#     Fully connected autoencoder for 2D input (profiles, depth).
    
#     Args:
#         input_shape (tuple): Tuple of (profiles, depth).
#         features_list (list): List of feature dimensions. Must satisfy features_list[0]==depth.
#         pooling (str): "Max", "Avg", or "None" for pooling over profiles.
#         pool_kernel (int): Kernel size for pooling.
#         act_fn_str (str): Activation function ("LeakyRelu", "Relu", etc.).
#         dropout_proba (float): Dropout probability.
#     """
#     def __init__(self,
#                  input_shape: tuple = (100, 64),
#                  features_list: list = [64, 32, 16],
#                  pooling: str = "None",
#                  pool_kernel: int = 2,
#                  act_fn_str: str = "LeakyRelu",
#                  dropout_proba: float = 0.0):
#         super().__init__()
#         self.input_profiles, self.input_depth = input_shape
#         self.pooling = pooling if pooling != "None" else None
#         self.pool_kernel = pool_kernel

#         # Activation function dictionary with LeakyRelu option
#         act_fn_dict = {
#             "LeakyRelu": nn.LeakyReLU(),
#             "Relu": nn.ReLU(),
#             "Sigmoid": nn.Sigmoid(),
#             "Tanh": nn.Tanh(),
#             "None": nn.Identity()
#         }
#         self.act_fn = act_fn_dict.get(act_fn_str, nn.LeakyReLU())

#         # Build encoder
#         encoder_layers = []
#         in_features = features_list[0]  # should match input depth
#         assert in_features == self.input_depth, "The first element of features_list must match the input depth."
#         for i in range(len(features_list)-1):
#             out_features = features_list[i+1]
#             encoder_layers.append(nn.Linear(in_features, out_features))
#             encoder_layers.append(self.act_fn)
#             if dropout_proba > 0:
#                 encoder_layers.append(nn.Dropout(p=dropout_proba))
#             # Apply pooling over the profiles dimension after each layer except the last encoder layer
#             if self.pooling and i != len(features_list)-2:
#                 encoder_layers.append(PoolProfiles(self.pooling, pool_kernel))
#             in_features = out_features
#         self.encoder = nn.Sequential(*encoder_layers)

#         # Build decoder (mirror of encoder)
#         decoder_layers = []
#         # If pooling was used in encoder, profiles dimension was reduced.
#         # Compute reduced profiles size after pooling layers in encoder.
#         reduced_profiles = self.input_profiles
#         if self.pooling:
#             # Apply pooling for each encoder layer that used it.
#             n_pool = len([1 for i in range(len(features_list)-1) if i != (len(features_list)-2)])
#             for _ in range(n_pool):
#                 reduced_profiles = (reduced_profiles - self.pool_kernel) // self.pool_kernel + 1
#             self.reduced_profiles = reduced_profiles
#             # Upsample back to original profiles before decoder.
#             self.upsample = UpsampleProfiles(self.input_profiles)
#         else:
#             self.upsample = None

#         rev_features = features_list[::-1]
#         in_features = rev_features[0]
#         for i in range(len(rev_features)-1):
#             out_features = rev_features[i+1]
#             decoder_layers.append(nn.Linear(in_features, out_features))
#             decoder_layers.append(self.act_fn)
#             if dropout_proba > 0:
#                 decoder_layers.append(nn.Dropout(p=dropout_proba))
#             in_features = out_features
#         self.decoder = nn.Sequential(*decoder_layers)

#     def forward(self, x):
#         # x shape: (batch, profiles, depth)
#         # Process each profile's depth vector independently:
#         # Flatten profiles and depth dimension across last axis using linear layers
#         # Note: Linear layers operate on last dimension.
#         x_enc = self.encoder(x)
#         if self.upsample:
#             x_enc = self.upsample(x_enc)
#         x_dec = self.decoder(x_enc)
#         return x_dec

# # ...existing code or tests...
