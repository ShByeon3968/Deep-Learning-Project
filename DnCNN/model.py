import torch.nn as nn
import torch.nn.init as init

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super(DnCNN,self).__init__()
        kernel_size = 3 
        padding = 1
        layers = []

        # 첫번째 레이어 설정
        layers.append(nn.Conv2d(in_channels=image_channels,
                                out_channels=n_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        # 중간 레이어들
        # 2 ~ (D-1)
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels,out_channels=n_channels,
                                    kernel_size=kernel_size,padding=padding,
                                    bias=False))
            layers.append(nn.BatchNorm2d(n_channels,eps=0.0001,momentum=0.95))
            layers.append(nn.ReLU(inplace=True))

        # 마지막 레이어
        layers.append(nn.Conv2d(n_channels,image_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)
        
    def _initialize_weights(self):
        # 가중치 초기화화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 출력 채널 수 기준으로 분산 계산
                # Relu 설정
                init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self,x):
        y = x
        out = self.dncnn(x)
        return y - out