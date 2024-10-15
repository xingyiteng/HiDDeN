import argparse
import os
import torch.nn
import torchvision.transforms.functional as TF
from PIL import Image
from SSIM import SSIM
import utils
from model.hidden import *
from noise_layers.noiser import Noiser
from noise_layers.crop import Crop


def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='mdfa/identity/options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', default='mdfa/identity/mdfa_no_noise_32--epoch-200.pyt', type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12, type=int, help='The batch size.')
    parser.add_argument('--source-dir', '-s', default='D:\\workspace\\watermark\\DataSet\\COCO\\data\\test', type=str,
                        help='The directory containing images to watermark')

    args = parser.parse_args()

    train_options, hidden_config, noise_config = utils.load_options(args.options_file)

    # 如果是组合噪声，需要重新修改noise_config，如果是单层噪声，注释以下代码
    noise_config = [Crop((0.4, 0.55), (0.4, 0.55))]
    # noise_config = [Cropout((0.55, 0.6), (0.55, 0.6))]
    # noise_config = [Dropout((0.55, 0.6))]
    # noise_config = [Resize((0.7, 0.8))]
    # noise_config = ['JpegPlaceholder']

    # 使用修改后的noise_config创建Noiser
    noiser = Noiser(noise_config, device)

    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)

    # 初始化累计值
    total_ber = 0
    total_psnr = 0
    total_ssim = 0
    image_count = 0

    # 遍历图片文件夹
    for filename in os.listdir(args.source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(args.source_dir, filename)
            image_pil = Image.open(image_path).convert('RGB')  # 确保图像是RGB模式

            # 调整图像大小为模型期望的尺寸
            expected_height = hidden_config.H
            expected_width = hidden_config.W
            image_pil = image_pil.resize((expected_width, expected_height), Image.BILINEAR)

            image_tensor = TF.to_tensor(image_pil).to(device)
            image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
            image_tensor.unsqueeze_(0)

            # 确保图像有3个通道
            if image_tensor.shape[1] != 3:
                image_tensor = image_tensor.repeat(1, 3, 1, 1)

            message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0],
                                                            hidden_config.message_length))).to(device)
            losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch([image_tensor, message])
            decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
            message_detached = message.detach().cpu().numpy()

            # BER计算
            ber = np.mean(decoded_rounded != message_detached)
            total_ber += ber

            # PSNR计算
            mse_loss = nn.MSELoss().to(device)
            g_loss_enc = mse_loss(encoded_images, image_tensor)
            psnr = 10 * torch.log10(4 / g_loss_enc)
            total_psnr += psnr.item()

            # SSIM计算
            ssim_loss = SSIM()
            g_loss_enc_ssim = ssim_loss(encoded_images, image_tensor)
            total_ssim += g_loss_enc_ssim.item()

            image_count += 1

    # 计算平均值
    avg_ber = total_ber / image_count
    avg_psnr = total_psnr / image_count
    avg_ssim = total_ssim / image_count

    print(f'Average Correct Bit Rate : {1 - avg_ber:.3f}')
    print(f'Average PSNR : {avg_psnr:.3f}')
    print(f'Average SSIM : {avg_ssim:.3f}')

if __name__ == '__main__':
    main()
