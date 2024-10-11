import torch.nn as nn

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out, stride=1):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            # 加速训练、减少梯度消失，梯度爆炸
            nn.BatchNorm2d(channels_out),
            # 引入非线性变换、拟合复杂的非线性关系
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)
