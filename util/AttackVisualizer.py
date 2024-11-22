import os
import torch
import numpy as np
from PIL import Image
import torchvision.utils as vutils
from noise_layers.crop import Crop
from noise_layers.cropout import Cropout
from noise_layers.dropout import Dropout
from noise_layers.resize import Resize
from noise_layers.jpeg_compression import JpegCompression


class AttackVisualizer:
    def __init__(self, save_path='attack_results'):
        """
        初始化攻击可视化器
        :param save_path: 结果保存的根目录
        """
        self.save_path = save_path
        self._create_dirs()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_dirs(self):
        """创建保存结果的目录结构"""
        for attack in ['crop', 'cropout', 'dropout', 'resize', 'jpeg']:
            os.makedirs(os.path.join(self.save_path, attack), exist_ok=True)

    def save_image(self, img_tensor, original_path, attack_type, attack_param):
        """
        保存单张图像，保持原始格式
        :param img_tensor: 攻击后的图像张量
        :param original_path: 原始图像路径
        :param attack_type: 攻击类型
        :param attack_param: 攻击参数
        """
        # 获取原始图像的格式
        original_format = original_path.split('.')[-1]
        # 获取原始文件名（不含路径和扩展名）
        original_name = os.path.basename(original_path).rsplit('.', 1)[0]
        # 构建新的文件名，包含攻击参数
        new_filename = f'{original_name}_{attack_type}_{attack_param}.{original_format}'
        save_path = os.path.join(self.save_path, attack_type, new_filename)

        # 将tensor转换为PIL图像并保存
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).clip(0, 255).astype('uint8')
        img_pil = Image.fromarray(img_np)
        img_pil.save(save_path, format=original_format)
        print(f"Saved attacked image to: {save_path}")

    def visualize_attacks(self, image_path):
        """
        对单张图像进行各种攻击并保存结果
        :param image_path: 原始图像的路径
        """
        print(f"\nProcessing image: {image_path}")

        # 读取图像并转换为tensor
        img = Image.open(image_path)
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        try:
            # Crop attack (50%)
            print("Applying Crop attack...")
            crop = Crop((0.5, 0.5), (0.5, 0.5))
            cropped = crop([img_tensor.clone(), img_tensor.clone()])[0]  # 使用原图作为cover
            self.save_image(cropped, image_path, 'crop', '50')
        except Exception as e:
            print(f"Error in Crop attack: {str(e)}")

        try:
            # Cropout attack (50%)
            print("Applying Cropout attack...")
            cropout = Cropout((0.5, 0.5), (0.5, 0.5))
            cropouted = cropout([img_tensor.clone(), img_tensor.clone()])  # 使用原图作为cover
            self.save_image(cropouted[0], image_path, 'cropout', '50')
        except Exception as e:
            print(f"Error in Cropout attack: {str(e)}")

        try:
            # Dropout attack (50%)
            print("Applying Dropout attack...")
            dropout = Dropout((0.5, 0.5)) # 修改为元组形式，表示保持率范围
            dropped = dropout([img_tensor.clone(), img_tensor.clone()])[0]  # 使用原图作为cover
            self.save_image(dropped, image_path, 'dropout', '50')
        except Exception as e:
            print(f"Error in Dropout attack: {str(e)}")

        try:
            # Resize attack (70%)
            print("Applying Resize attack...")
            resize = Resize((0.7, 0.7))  # 修改为元组形式
            resized = resize([img_tensor.clone(), img_tensor.clone()])[0]  # 使用原图作为cover
            self.save_image(resized, image_path, 'resize', '70')
        except Exception as e:
            print(f"Error in Resize attack: {str(e)}")

        try:
            # JPEG attack (quality=50)
            print("Applying JPEG compression...")
            jpeg = JpegCompression(self.device)
            jpeged = jpeg([img_tensor.clone(), img_tensor.clone()])[0]  # 使用原图作为cover
            self.save_image(jpeged, image_path, 'jpeg', 'q50')
        except Exception as e:
            print(f"Error in JPEG attack: {str(e)}")

        print("\nAll attacks completed!")


def main():
    """
    主函数，展示如何使用AttackVisualizer
    """
    # 设置保存路径
    save_path = 'attack_results'

    # 创建可视化器实例
    visualizer = AttackVisualizer(save_path=save_path)

    # 测试图像路径
    test_image_path = 'D:\\workspace\\\watermark\\\HiDDeN\\test.png'  # 替换为你的测试图像路径

    # 确保测试图像存在
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        return

    # 执行攻击可视化
    try:
        visualizer.visualize_attacks(test_image_path)
        print(f"\nResults have been saved to: {save_path}")
        print("Directory structure:")
        print("attack_results/")
        print("├── crop/")
        print("├── cropout/")
        print("├── dropout/")
        print("├── resize/")
        print("└── jpeg/")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == '__main__':
    main()