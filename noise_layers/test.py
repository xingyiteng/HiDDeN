import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from noise_layers.crop import Crop
from noise_layers.dropout import Dropout
from noise_layers.cropout import Cropout
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.resize import Resize


class NoiseAttackTester:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        self.to_pil = transforms.ToPILImage()

    def check_tensor(self, tensor, name="tensor"):
        """检查tensor的有效性"""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
        if not tensor.is_floating_point():
            tensor = tensor.float()
        if tensor.dim() not in [3, 4]:
            raise ValueError(f"{name} must have 3 or 4 dimensions, got {tensor.dim()}")
        return tensor

    def resize_tensor(self, tensor, target_size):
        """调整tensor大小的辅助函数"""
        tensor = self.check_tensor(tensor)
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        return resized.squeeze(0)

    def generate_difference_map(self, original, noised, original_size):
        """生成差异图"""
        # 检查输入
        original = self.check_tensor(original, "original")
        noised = self.check_tensor(noised, "noised")

        # 确保是3D tensor (C, H, W)
        if original.dim() == 4:
            original = original.squeeze(0)
        if noised.dim() == 4:
            noised = noised.squeeze(0)

        # 确保大小一致
        if noised.shape[-2:] != original_size:
            noised = self.resize_tensor(noised, original_size)

        # 确保通道数正确
        if original.shape[0] != 3 or noised.shape[0] != 3:
            raise ValueError("Both tensors must have 3 channels (RGB)")

        # 转换为灰度图
        original_gray = 0.299 * original[0] + 0.587 * original[1] + 0.114 * original[2]
        noised_gray = 0.299 * noised[0] + 0.587 * noised[1] + 0.114 * noised[2]

        # 计算差异
        diff = torch.abs(original_gray - noised_gray)

        # 归一化，确保不会除以0
        max_val = diff.max()
        if max_val > 0:
            diff = diff / max_val

        return diff

    def attack_and_save(self, image_path, save_path_noised, save_path_diff, attack_type='crop'):
        """对图像进行攻击并保存攻击后的图像和差异图"""
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # 加载图像
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        original_size = (img_tensor.shape[2], img_tensor.shape[3])

        # 根据图片中的参数设置攻击
        try:
            if attack_type == 'crop':
                noise_layer = Crop(height_ratio_range=(1, 1), width_ratio_range=(1, 1))
                attacked = noise_layer([img_tensor])[0]
                attacked = self.resize_tensor(attacked, original_size)
            elif attack_type == 'dropout':
                noise_layer = Dropout(keep_ratio_range=(0.2, 0.2))
                attacked = noise_layer([img_tensor, img_tensor])[0]
                attacked = attacked.squeeze(0)  # 修复维度问题
            elif attack_type == 'cropout':
                noise_layer = Cropout(height_ratio_range=(0.2, 0.2), width_ratio_range=(0.2, 0.2))
                attacked = noise_layer([img_tensor, img_tensor])[0]
                attacked = attacked.squeeze(0)  # 修复维度问题
            elif attack_type == 'jpeg':
                noise_layer = JpegCompression(device=self.device, quality=50)  # 修改参数名
                attacked = noise_layer([img_tensor])[0]
            else:
                raise ValueError(f"Unsupported attack type: {attack_type}")
        except Exception as e:
            raise RuntimeError(f"Error during {attack_type} attack: {str(e)}")

        # 确保attacked是有效的tensor
        attacked = self.check_tensor(attacked)

        try:
            # 生成差异图
            diff_map = self.generate_difference_map(img_tensor, attacked, original_size)

            # 保存攻击后的图像
            attacked = attacked.cpu().clamp(0, 1)
            if attacked.dim() == 4:
                attacked = attacked.squeeze(0)  # 确保是3D tensor
            attacked_img = self.to_pil(attacked)
            attacked_img.save(save_path_noised)

            # 保存差异图
            diff_map = diff_map.cpu()
            diff_img = self.to_pil(diff_map.unsqueeze(0))
            diff_img.save(save_path_diff)

            print(f"Successfully saved attacked image to {save_path_noised}")
            print(f"Successfully saved difference map to {save_path_diff}")

        except Exception as e:
            raise RuntimeError(f"Error saving images: {str(e)}")


# 使用示例
if __name__ == '__main__':
    tester = NoiseAttackTester()

    # 测试所有攻击类型
    attack_types = ['crop', 'cropout', 'dropout', 'jpeg']

    for attack_type in attack_types:
        try:
            tester.attack_and_save(
                image_path='D:/workspace/watermark/HiDDeN/test.png',
                save_path_noised=f'attacked_{attack_type}.png',
                save_path_diff=f'diff_{attack_type}.png',
                attack_type=attack_type
            )
        except Exception as e:
            print(f"Error processing {attack_type} attack: {str(e)}")