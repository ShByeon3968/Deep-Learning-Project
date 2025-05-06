import torch
import torch.nn as nn

def patch_partition(x:torch.Tensor, window_size:int):
    '''
    Args:
    x: (B,H,W,C) - 입력 feature Map
    window_size: 윈도우 크기
    return:
        windows : (num_windows * B, window_size, window_size,C)
    '''

    B,H,W,C = x.shape

    # H,W 차원을 window 단위로 reshape
    x = x.view(
        B,
        H // window_size, window_size,
        W // window_size, window_size,
        C
    )

    # 윈도우를 모양에 맞게 
    # congiguous : Tensor의 각 값들이 메모리에도 순차적으로 저장되어 있는지 여부
    windows = x.permute(0,1,3,2,4,5).contiguous().view(
        -1, window_size,window_size,C
    )
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: 윈도우 크기
        H, W: 원래 height, width
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B,
        H // window_size, W // window_size,
        window_size, window_size, -1
    )

    # 윈도우를 원래 이미지 순서로 재배열
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    def __init__(self,dim,input_resolution,num_heads,
                 window_size=7,shift_size=0,mlp_ratio=4.,drop_out=0.1):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim # 입력/출력 채널 수
        self.input_resolution = input_resolution # 입력 Feature map의 해상도
        self.window_size = window_size # 윈도우 크기
        self.shift_size = shift_size # Shifted Window 일 경우 이동 크기 (보통 window_size//2)
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,num_heads,dropout=drop_out,batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim,int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(int(dim*mlp_ratio),dim),
            nn.Dropout(drop_out)
        )

    def forward(self,x:torch.Tensor):
        '''
        x : (B, H*W,C)
        shift_size = 0: W-MSA
        shift_size > 0: SW-MSA
        '''
        H,W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        
        if self.shift_size > 0:
            # X를 오른쪽 아래로 이동
            #  torch.roll 은 원하는 dim으로 tensor를 이동
            shifted_x = torch.roll(x,shifts=(-self.shift_size,-self.shift_size),dims=(1,2))
        else:
            shifted_x = x

        # 윈도우 단위로 나누기
        x_windows = patch_partition(shifted_x,self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 윈도우 단위로 MultiheadAttention 수행
        shortcut = x_windows
        x_windows = self.norm1(x_windows)
        attn_windows ,_ = self.attn(x_windows,x_windows,x_windows)
        x_windows = shortcut + attn_windows # residual connection
        
        # 윈도우 병합
        x_windows = x_windows.view(-1, self.window_size , self.window_size, C)
        x_windows = window_reverse(x_windows, self.window_size, H, W)

        # x를 원래 위치로 되돌리기
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # MLP 수행
        x = x + self.mlp(self.norm2(x))
        return x
