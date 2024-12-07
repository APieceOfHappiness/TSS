import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, inter_channels, kernel_size, dilations):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.conv_list_1 = nn.ModuleList([
            nn.ConvTranspose1d(in_channels=inter_channels,
                               out_channels=inter_channels,
                               kernel_size=kernel_size,
                               dilation=dilation[0],
                               padding=dilation[0] * (kernel_size - 1) // 2)
            for dilation in dilations
        ])

        self.conv_list_2 = nn.ModuleList([
            nn.ConvTranspose1d(in_channels=inter_channels,
                               out_channels=inter_channels,
                               kernel_size=kernel_size,
                               dilation=dilation[1],
                               padding=dilation[1] * (kernel_size - 1) // 2)
            for dilation in dilations 
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.conv_list_1)):
            out = self.leaky_relu(x)
            out = self.conv_list_1[i](out)

            out = self.leaky_relu(out)
            out = self.conv_list_2[i](out)

            out = out + x
        return  out


class MRF(nn.Module):
    def __init__(self, inter_channels, kernel_sizes, dilations):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.res_blocks = nn.ModuleList([
            ResBlock(inter_channels=inter_channels,
                     kernel_size=kernel_size,
                     dilations=dilations)
            for kernel_size in kernel_sizes
        ])

    @property
    def size(self) -> int:
        return len(self.kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for kernel_idx in range(self.size):
            out += self.res_blocks[kernel_idx](out)
        return out / self.size

class Generator(nn.Module):
    def __init__(self, kernel_sizes: list[int], hidden_dim: int, res_block_config: dict[str]) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.layers_cnt = len(self.kernel_sizes)

        self.feature_expansion = nn.Conv1d(in_channels=80, out_channels=hidden_dim, 
                                           kernel_size=7, stride=1, padding=(7 - 1) // 2)
        self.feature_reduction = nn.Conv1d(in_channels=hidden_dim // 2 ** (self.layers_cnt), out_channels=1, 
                                           kernel_size=7, stride=1, padding=(7 - 1) // 2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        self.upscale_list = nn.ModuleList([
            nn.ConvTranspose1d(in_channels=self.hidden_dim // 2 ** (kernel_idx), 
                               out_channels=self.hidden_dim // 2 ** (kernel_idx + 1),
                               kernel_size=kernel_size,
                               stride=kernel_size // 4)
            for kernel_idx, kernel_size in enumerate(self.kernel_sizes)
        ])
        
        self.mrf_list = nn.ModuleList([
            MRF(inter_channels=self.hidden_dim // 2 ** (idx + 1),
                kernel_sizes=res_block_config['kernel_sizes'],
                dilations=res_block_config['dilations'])
            for idx in range(self.layers_cnt)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape [B, T0, D]
        Returns:
            out: output tensor of shape [B, T1, 1] - wav
        """

        out = self.feature_expansion(x)
        for up_kernel_idx in range(self.layers_cnt):
            # upscaling
            out = self.leaky_relu(out)
            out = self.upscale_list[up_kernel_idx](out)

            # MRF
            out = self.mrf_list[up_kernel_idx](out)

        out = self.leaky_relu(out)
        out = self.feature_reduction(out)
        out = torch.tanh(out)
        return out
    

# class ResBlock(nn.Module):  # TODO: rename, because it is not a resblock from the paper
#     def __init__(self, channels, kernel_size, dilation):
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
#         self.conv_1d = nn.ConvTranspose1d(in_channels=channels, 
#                                             out_channels=channels,
#                                             kernel_size=kernel_size,
#                                             dilation=dilation,
#                                             padding=dilation * (kernel_size - 1) // 2)
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.leaky_relu(x)
#         out = self.conv_id(out)
#         return out

# class ResMatrix(nn.Module):
#     def __init__(self, kernel_sizes: list[int], dilations: list[list[int]], **res_block_config) -> None:
#         super().__init__()
#         self.res_matrix = nn.ModuleList([
#             nn.ModuleList([
#                 ResBlock(kernel_size=kernel_size, dilation=dilation, **res_block_config)
#                 for dilation in dilations
#             ])
#             for kernel_size in kernel_sizes
#         ])

#     def forward(self, x: torch.Tensor) -> None:
#         raise RuntimeError('This class was created as an analogue of ModuleList and it does not have a forward method')

#     def __getitem__(self, idx: int) -> nn.ModuleList:
#         return self.res_matrix[idx]
    
#     def shape(self) -> tuple[int]:
#         if len(self.res_matrix) == 0:
#             return (0, 0)
#         return (len(self.res_matrix), len(self.res_matrix[0]))