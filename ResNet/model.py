import torch
import torch.nn as nn
from typing import List, Optional,Any

class Module(torch.nn.Module):
    r"""
    Wraps ``torch.nn.Module`` to overload ``__call__`` instead of
    ``forward`` for better type checking.
    
    `PyTorch Github issue for clarification <https://github.com/pytorch/pytorch/issues/44605>`_
    """

    def _forward_unimplemented(self, *input: Any) -> None:
        # To stop PyTorch from giving abstract methods warning
        pass

    def __init_subclass__(cls, **kwargs):
        if cls.__dict__.get('__call__', None) is None:
            return

        setattr(cls, 'forward', cls.__dict__['__call__'])
        delattr(cls, '__call__')

    @property
    def device(self):
        params = self.parameters()
        try:
            sample_param = next(params)
            return sample_param.device
        except StopIteration:
            raise RuntimeError(f"Unable to determine"
                               f" device of {self.__class__.__name__}") from None

## Shortcut connection을 위한 Linear Projection
class ShortcutProjection(Module):
    def __init__(self, in_channels:int, out_channels:int, stride:int):
        '''
        in_channels : 채널 입력 수
        out_channels : F(x,{W})의 채널 수
        '''
        super().__init__()
        # Convolution layer
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x:torch.Tensor):
        return self.bn(self.conv(x))
    
# Residual Block
class ResidualBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        # 첫번째 3x3 Convolution
        self.conv1 = nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 두번째 3x3 Convolution
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # activation
        self.act = nn.ReLU()

        # Stride의 길이가 1이 아니거나 입력 채널수와 출력 채널수가 다를 때
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels,out_channels,stride)
        else:
            self.shortcut = nn.Identity() #입력값을 그대로 출력값으로 반환하는 모듈
    
    def forward(self,x:torch.Tensor):
        shortcut = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + shortcut)
    
# Bottleneck
class BottleneckResidualBlock(Module):
    def __init__(self, in_channels: int,bottleneck_channels:int ,out_channels: int, stride: int):
        super().__init__()
        # 1x1 convolution
        self.conv1 = nn.Conv2d(in_channels,bottleneck_channels,kernel_size=1,stride=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(bottleneck_channels,out_channels,kernel_size=1,stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()
        
        self.act3 = nn.ReLU()

    def forward(self,x:torch.Tensor):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.act3(x + shortcut)

class ResNetBase(Module):
    def __init__(self,n_blocks:List[int], n_channels:List[int],bottlenecks:Optional[List[int]]=None,
                 img_channels:int = 3, first_kernel_size:int=7):
        
        super().__init__()
        # 블록 수와 각 채널별 피처맵의 사이즈
        assert len(n_blocks) == len(n_channels)
        assert bottlenecks is None or len(bottlenecks) == len(n_channels)

        self.conv = nn.Conv2d(img_channels, n_channels[0],kernel_size=first_kernel_size, stride=2, padding=first_kernel_size // 2)
        self.bn = nn.BatchNorm2d(n_channels[0])
        blocks = []
        prev_channels = n_channels[0]
        for i, channels in enumerate(n_channels):
            stride = 2 if len(blocks) == 0 else 1
            if bottlenecks is None:
                blocks.append(ResidualBlock(prev_channels, channels, stride=stride))
            else:
                blocks.append(BottleneckResidualBlock(prev_channels, bottlenecks[i], channels,stride=stride))
            prev_channels = channels
            for _ in range(n_blocks[i] - 1):
                if bottlenecks is None:
                    blocks.append(ResidualBlock(channels,channels,stride=1))
                else:
                    blocks.append(BottleneckResidualBlock(channels, bottlenecks[i], channels, stride=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self,x:torch.Tensor):
        x = self.bn(self.conv(x))
        x = self.blocks(x)
        x = x.view(x.shape[0],x.shape[1],-1)
        return x.mean(dim=1)

