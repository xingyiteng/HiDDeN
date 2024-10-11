import torch
import torch.nn as nn
from model.conv_bn_relu import ConvBNRelu
from model.mdfa import MDFA
from options import HiDDenConfiguration


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H  # 指定输入图像高度
        self.W = config.W  # 指定输入图像宽度
        self.conv_channels = config.encoder_channels  # 卷积通道数
        self.num_blocks = config.encoder_blocks  # 卷积块块数
        layers = [ConvBNRelu(3, self.conv_channels)]
        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)
        # 包含4个ConvBNRelu块
        self.conv_layers = nn.Sequential(*layers)

        # Ico：3通道 + 特征图：64通道 + message: 30长度  => 64通道
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)

        # 64通道 => 3通道 图片大小不变
        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

        # MDFA
        self.mdfa = MDFA(dim_in=(self.conv_channels + 3 + config.message_length), dim_out=self.conv_channels)

    def forward(self, image, message):
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        # expanded_message 将被扩展为形状 (batch_size, message_length, self.H, self.W)
        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)

        # 特征图
        encoded_image = self.conv_layers(image)
        # concatenate expanded message and image
        # concat 的形状将是 (batch_size, message_length + self.conv_channels + 3, self.H, self.W)。
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)

        im_w = self.mdfa(concat)
        # im_w = self.after_concat_layer(concat)

        im_w = self.final_layer(im_w)
        return im_w
