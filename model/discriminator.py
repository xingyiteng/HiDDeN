import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu

class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Discriminator, self).__init__()

        layers = [ConvBNRelu(3, config.discriminator_channels)]
        for _ in range(config.discriminator_blocks-1):
            layers.append(ConvBNRelu(config.discriminator_channels, config.discriminator_channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(config.discriminator_channels, 1)

    def forward(self, image):
        X = self.before_linear(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        # X = torch.sigmoid(X)
        return X

if __name__ == '__main__':
    from torchinfo import summary
    from options import HiDDenConfiguration

    # 创建配置对象
    config = HiDDenConfiguration(
        H=128,
        W=128,
        encoder_channels=64,
        encoder_blocks=4,
        decoder_channels=64,
        decoder_blocks=7,
        message_length=30,
        use_discriminator=True,
        use_vgg=False,
        discriminator_channels=64,
        discriminator_blocks=3,
        decoder_loss=1,
        encoder_loss=1,
        adversarial_loss=1
    )

    # 创建模型并移至设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Discriminator(config).to(device)

    # 使用torchinfo的summary
    summary(
        model,
        input_size=[(1, 3, 128, 128)],  # 判别器只需要图像输入
        device=device,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
    )