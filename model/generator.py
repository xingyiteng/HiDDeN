import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()

        encoder_lis = [
            # 3*128*128
            nn.Conv2d(gen_input_nc, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 64*64*64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 128*32*32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            # 256*16*16
        ]

        bottle_neck_lis = [ResnetBlock(256),
                           ResnetBlock(256),
                           ResnetBlock(256),
                           ResnetBlock(256),]

        decoder_lis = [
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 128*32*32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 64*64*64
            nn.ConvTranspose2d(64, image_nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # 3*128*128
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x

# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


def main():
    # 设置随机种子以确保结果可重现
    torch.manual_seed(0)

    # 定义输入参数
    batch_size = 32
    gen_input_nc = 3  # 输入通道数
    image_nc = 3  # 输出通道数
    image_size = 128  # 图像大小

    # 创建Generator实例
    generator = Generator(gen_input_nc, image_nc)

    # 生成随机输入数据
    random_input = torch.randn(batch_size, gen_input_nc, image_size, image_size)

    # 将数据传递给generator
    with torch.no_grad():  # 不计算梯度
        output = generator(random_input)

    # 打印输入和输出的形状
    print(f"Input shape: {random_input.shape}")
    print(f"Output shape: {output.shape}")

    # 检查输出值是否在[-1, 1]范围内（因为使用了Tanh激活函数）
    print(f"Output min value: {output.min().item()}")
    print(f"Output max value: {output.max().item()}")


if __name__ == "__main__":
    main()