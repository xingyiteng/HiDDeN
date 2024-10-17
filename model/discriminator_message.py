import torch.nn as nn
import torch


class discriminator_message(nn.Module):
    def __init__(self, input_dim=30):
        super(discriminator_message, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def main():
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)

    # 创建Discriminator实例
    discriminator = discriminator_message()

    # 生成随机输入
    batch_size = 32
    input_dim = 30
    random_input = torch.randint(0, 2, (batch_size, input_dim), dtype=torch.float32)

    # 打印随机输入的一部分
    print("随机输入样本（前5个）：")
    print(random_input[:5])

    # 使用模型进行预测
    with torch.no_grad():
        output = discriminator(random_input)

    # 打印输出
    print("\n模型输出：")
    print(output)

    # 打印输出的形状
    print("\n输出形状：", output.shape)

    # 检查输出范围
    print("\n输出最小值：", output.min().item())
    print("输出最大值：", output.max().item())


if __name__ == "__main__":
    main()
