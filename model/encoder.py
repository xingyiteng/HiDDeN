import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


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
        self.conv_layers = nn.Sequential(*layers)

        # 输入图像3通道 卷积后的特征图conv_channels通道 消息的长度message_length
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)

        # 通过将高维特征映射变换为RGB图像 空间分辨率保持不变
        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, image, message):

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly

        # 将张量 message 在最后一个维度（-1 表示最后一个维度）上添加一个维度
        # 使得 message 的维度数增加 1。
        # 具体来说，message 是一个 PyTorch 张量，它可能具有形状 (batch_size, message_length)。
        # 在这里，message_length 是消息的长度，batch_size 是批次的大小。
        # 通过执行 message.unsqueeze(-1)，将在最后一个维度上添加一个大小为 1 的新维度
        # 使 message 的形状变为 (batch_size, message_length, 1)。
        expanded_message = message.unsqueeze(-1)
        # 原地操作，添加1个维度
        expanded_message.unsqueeze_(-1)

        # expanded_message 将被扩展为形状 (batch_size, message_length, self.H, self.W)
        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)

        # 特征图
        encoded_image = self.conv_layers(image)
        # concatenate expanded message and image
        # 将三个张量 expanded_message, encoded_image, 和 image 沿着第一个维度（dim=1）进行拼接（concatenate）
        # 生成一个包含它们所有内容的新张量。

        # expanded_message 是包含嵌入消息的张量，通常用于将消息嵌入到图像中。它的形状是 (batch_size, message_length, self.H, self.W)。
        # encoded_image 是经过编码的图像特征的张量，形状为 (batch_size, self.conv_channels, self.H, self.W)。
        # image 是原始输入图像的张量，形状为 (batch_size, 3, self.H, self.W)，其中 3 表示图像具有三个通道（RGB）。

        # concat 的形状将是 (batch_size, message_length + self.conv_channels + 3, self.H, self.W)。
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)

        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w
