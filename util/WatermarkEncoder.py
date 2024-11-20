import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from typing import Union, List, Tuple
import utils
from model.hidden import Hidden
from noise_layers.noiser import Noiser


class WatermarkEncoder:
    """图像水印编码工具类，用于批量处理图像的水印嵌入"""

    def __init__(self, options_file: str, checkpoint_file: str, device: torch.device):
        """
        初始化水印编码器
        Args:
            options_file: 配置文件路径
            checkpoint_file: 模型检查点文件路径
            device: 计算设备（CPU/GPU）
        """
        self.device = device
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}

        # 加载配置和模型
        train_options, hidden_config, noise_config = utils.load_options(options_file)
        self.hidden_config = hidden_config

        # 创建noiser
        self.noiser = Noiser(noise_config, device)

        # 加载模型
        checkpoint = torch.load(checkpoint_file, map_location=device)
        self.hidden_net = Hidden(hidden_config, device, self.noiser, None)
        utils.model_from_checkpoint(self.hidden_net, checkpoint)

        # 生成固定的随机水印信息
        np.random.seed(42)  # 设置随机种子确保水印信息一致
        self.watermark = torch.Tensor(
            np.random.choice([0, 1], (1, hidden_config.message_length))
        ).to(device)

    def generate_difference_image(self, original_tensor: torch.Tensor, encoded_tensor: torch.Tensor) -> Image.Image:
        """
        生成原始图像和编码图像的差值图像
        Args:
            original_tensor: 原始图像tensor
            encoded_tensor: 编码后的图像tensor
        Returns:
            Image.Image: 差值图像
        """
        # 转换为灰度图像
        original_gray = 0.299 * original_tensor[0] + 0.587 * original_tensor[1] + 0.114 * original_tensor[2]
        encoded_gray = 0.299 * encoded_tensor[0] + 0.587 * encoded_tensor[1] + 0.114 * encoded_tensor[2]

        # 计算差值
        diff = torch.abs(encoded_gray - original_gray)

        # 增强差异（可选，使差异更明显）
        diff = diff * 5  # 放大差异
        diff = diff.clamp(0, 1)  # 确保值在[0,1]范围内

        # 转换为PIL图像
        diff_image = TF.to_pil_image(diff)
        return diff_image

    def process_single_image(self, image_path: str) -> bool:
        """
        处理单张图像的水印嵌入
        Args:
            image_path: 图像路径
        Returns:
            bool: 处理是否成功
        """
        try:
            # 读取并预处理图像
            image_pil = Image.open(image_path).convert('RGB')

            # 调整图像大小为模型期望的尺寸
            image_pil = image_pil.resize((self.hidden_config.W, self.hidden_config.H), Image.BILINEAR)

            # 转换为tensor
            image_tensor = TF.to_tensor(image_pil).to(self.device)
            image_tensor_orig = image_tensor.clone()  # 保存原始图像tensor
            image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
            image_tensor.unsqueeze_(0)

            # 确保图像有3个通道
            if image_tensor.shape[1] != 3:
                image_tensor = image_tensor.repeat(1, 3, 1, 1)

            # 使用编码器嵌入水印
            with torch.no_grad():
                losses, (encoded_images, noised_images, decoded_messages) = self.hidden_net.validate_on_batch(
                    [image_tensor, self.watermark])

            # 后处理编码后的图像
            encoded_image = encoded_images.squeeze(0)  # 移除batch维度
            encoded_image = (encoded_image + 1) / 2  # 转换回[0, 1]范围
            encoded_image = encoded_image.clamp(0, 1)

            # 生成差值图像
            diff_image = self.generate_difference_image(image_tensor_orig.cpu(), encoded_image.cpu())

            # 构造文件名
            directory = os.path.dirname(image_path)
            filename = os.path.basename(image_path)
            filename_without_ext = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]

            # 编码图像的文件名
            encoded_filename = f'encoded_{filename}'
            # 差值图像的文件名
            diff_filename = f'diff_{filename_without_ext}.png'  # 使用PNG格式保存差值图像

            # 构造完整的保存路径
            encoded_path = os.path.join(directory, encoded_filename)
            diff_path = os.path.join(directory, diff_filename)

            # 保存图像
            encoded_pil = TF.to_pil_image(encoded_image.cpu())
            encoded_pil.save(encoded_path, quality=95)
            diff_image.save(diff_path)

            return True

        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")
            return False

    def process_directory(self, directory_path: str) -> Tuple[int, int]:
        """
        处理整个文件夹中的图像
        Args:
            directory_path: 文件夹路径
        Returns:
            Tuple[int, int]: (成功处理数量, 失败处理数量)
        """
        if not os.path.exists(directory_path):
            raise ValueError(f"目录 {directory_path} 不存在")

        success_count = 0
        failed_count = 0

        # 遍历文件夹中的所有文件
        for filename in os.listdir(directory_path):
            # 检查文件扩展名
            if any(filename.lower().endswith(ext) for ext in self.supported_formats):
                # 跳过已经处理过的图像
                if filename.startswith('encoded_'):
                    continue

                image_path = os.path.join(directory_path, filename)

                # 处理图像
                if self.process_single_image(image_path):
                    success_count += 1
                    print(f"成功处理图像: {filename}")
                else:
                    failed_count += 1
                    print(f"处理图像失败: {filename}")

        return success_count, failed_count


# 使用示例
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 配置文件路径
    options_file = 'D:\\workspace\\watermark\\HiDDeN\\experiments\\combined-noise\\options-and-config.pickle'
    checkpoint_file = 'D:\\workspace\\watermark\\HiDDeN\\experiments\\combined-noise\\checkpoints\\combined-noise--epoch-400.pyt'

    # 创建水印编码器实例
    watermark_encoder = WatermarkEncoder(
        options_file=options_file,
        checkpoint_file=checkpoint_file,
        device=device
    )

    # 指定要处理的文件夹路径
    folder_path = "D:\\workspace\\watermark\\HiDDeN\\image"

    try:
        # 处理文件夹中的所有图像
        success, failed = watermark_encoder.process_directory(folder_path)

        # 打印处理结果
        print(f"\n处理完成！")
        print(f"成功处理: {success} 张图像")
        print(f"处理失败: {failed} 张图像")
        print(f"使用的水印信息: {watermark_encoder.watermark.cpu().numpy()}")

    except Exception as e:
        print(f"发生错误: {str(e)}")