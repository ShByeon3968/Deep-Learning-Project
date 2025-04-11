import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size[0] - 1) * (2*window_size[1] -1),num_heads)
        )

class ShallowFeatureModule(nn.Module):
    def __init__(self,in_channels:int=3, out_channels:int=96):
        super(ShallowFeatureModule,self).__init__()
        self.msfe = nn.Conv2d(in_channels,out_channels)

    def forward(self,x):
        return self.msfe(x)
    

class UNetFeatureExtractionModule(nn.Module):
    def __init__(self):
        super(UNetFeatureExtractionModule,self).__init__()
        '''
        where MUFE(.) is the UNet architecture with Swin Transformer Block, which contains 8 Swin Transformer Layers in
        single block to replace the convolutions. The Swin Transformer Block (STB) and Swin Transformer Layer (STL) will
        be illustrated with details in next subsection.

        '''