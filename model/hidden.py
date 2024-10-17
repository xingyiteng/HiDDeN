import numpy as np
import torch
import torch.nn as nn

from SSIM import SSIM
from options import HiDDenConfiguration
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from vgg_loss import VGGLoss
from noise_layers.noiser import Noiser
from model.discriminator_message import discriminator_message
import torch.nn.functional as F


class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser: Noiser, tb_logger):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        :param noiser: Object representing stacked noise layers.
        :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
        """
        super(Hidden, self).__init__()

        self.encoder_decoder = EncoderDecoder(configuration, noiser).to(device)
        self.discriminator = Discriminator(configuration).to(device)
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters(), eps=1e-4)
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters(), eps=1e-4)
        self.discriminator_message = discriminator_message().to(device)
        self.optimizer_discrim_message = torch.optim.Adam(self.discriminator_message.parameters(), lr=1e-4)

        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False)
            self.vgg_loss.to(device)
        else:
            self.vgg_loss = None

        self.config = configuration
        self.device = device
        # SSIM
        self.ssim_loss = SSIM()
        # 二元交叉熵
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        self.ssim_weight = 0.1
        self.discriminator_message_weight = 0.2
        # Defined the labels used for training the discriminator/adversarial loss
        self.cover_label = 1
        self.encoded_label = 0

        self.tb_logger = tb_logger
        if tb_logger is not None:
            from tensorboard_logger import TensorBoardLogger
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            encoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/encoder_out'))
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            decoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/decoder_out'))
            discrim_final = self.discriminator._modules['linear']
            discrim_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/discrim_out'))

    def train_on_batch(self, batch: list):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        images, messages = batch

        batch_size = images.shape[0]
        self.encoder_decoder.train()
        self.discriminator.train()
        self.discriminator_message.train()
        with torch.enable_grad():
            # 清除所有优化器的梯度
            self.optimizer_enc_dec.zero_grad()
            self.optimizer_discrim.zero_grad()
            self.optimizer_discrim_message.zero_grad()

            # 前向传播
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            # 计算所有损失
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device).float()
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device).float()
            d_target_label_real = torch.ones((batch_size, 1), device=self.device)
            d_target_label_fake = torch.zeros((batch_size, 1), device=self.device)

            # 图像判别器损失
            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            # 消息判别器损失
            d_on_real_messages = self.discriminator_message(messages)
            # d_loss_on_real = self.bce_with_logits_loss(d_on_real_messages, d_target_label_real)
            d_loss_on_real = F.binary_cross_entropy_with_logits(d_on_real_messages, d_target_label_real)
            d_on_decoded = self.discriminator_message(decoded_messages.detach())
            # d_loss_on_decoded = self.bce_with_logits_loss(d_on_decoded, d_target_label_fake)
            d_loss_on_decoded = F.binary_cross_entropy_with_logits(d_on_decoded, d_target_label_fake)

            # 生成器损失
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, d_target_label_cover)
            g_loss_enc = self.mse_loss(encoded_images, images) if self.vgg_loss is None else self.mse_loss(
                self.vgg_loss(images), self.vgg_loss(encoded_images))
            g_loss_enc_ssim = self.ssim_loss(encoded_images, images)
            g_loss_dec = self.mse_loss(decoded_messages, messages)
            d_on_decoded_for_gen = self.discriminator_message(decoded_messages)
            # g_loss_d = self.bce_with_logits_loss(d_on_decoded_for_gen, d_target_label_real)
            g_loss_d = F.binary_cross_entropy_with_logits(d_on_decoded_for_gen, d_target_label_real)

            # 计算总损失
            d_loss = d_loss_on_cover + d_loss_on_encoded + d_loss_on_real + d_loss_on_decoded
            g_loss = self.config.adversarial_loss * g_loss_adv + self.ssim_weight * (1 - g_loss_enc_ssim) + \
                     self.config.encoder_loss * g_loss_enc + self.config.decoder_loss * g_loss_dec + \
                     g_loss_d * self.discriminator_message_weight

            # 反向传播
            d_loss.backward()
            g_loss.backward()

            # 更新参数
            self.optimizer_discrim.step()
            self.optimizer_discrim_message.step()
            self.optimizer_enc_dec.step()

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item(),
            'encoded_ssim   ': g_loss_enc_ssim.item(),
            'discrim_message_real': d_loss_on_real.item(),
            'discrim_message_fake': d_loss_on_decoded.item(),
            'message_bce    ': g_loss_d.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def validate_on_batch(self, batch: list):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        # if TensorboardX logging is enabled, save some of the tensors.
        if self.tb_logger is not None:
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)
            discrim_final = self.discriminator._modules['linear']
            self.tb_logger.add_tensor('weights/discrim_out', discrim_final.weight)

        images, messages = batch
        batch_size = images.shape[0]

        self.encoder_decoder.eval()
        self.discriminator.eval()
        self.discriminator_message.eval()

        with torch.no_grad():
            # 设置目标标签
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device).float()
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device).float()
            d_target_label_real = torch.ones((batch_size, 1), device=self.device)
            d_target_label_fake = torch.zeros((batch_size, 1), device=self.device)

            # 前向传播
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            # 图像判别器损失
            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            d_on_encoded = self.discriminator(encoded_images)
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            # 消息判别器损失
            d_on_real_messages = self.discriminator_message(messages)
            d_loss_on_real = F.binary_cross_entropy_with_logits(d_on_real_messages, d_target_label_real)
            d_on_decoded = self.discriminator_message(decoded_messages)
            d_loss_on_decoded = F.binary_cross_entropy_with_logits(d_on_decoded, d_target_label_fake)

            # 生成器损失
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, d_target_label_cover)
            g_loss_enc = self.mse_loss(encoded_images, images) if self.vgg_loss is None else self.mse_loss(
                self.vgg_loss(images), self.vgg_loss(encoded_images))
            g_loss_enc_ssim = self.ssim_loss(encoded_images, images)
            g_loss_dec = self.mse_loss(decoded_messages, messages)
            d_on_decoded_for_gen = self.discriminator_message(decoded_messages)
            g_loss_d = F.binary_cross_entropy_with_logits(d_on_decoded_for_gen, d_target_label_real)

            # 计算总损失
            d_loss = d_loss_on_cover + d_loss_on_encoded + d_loss_on_real + d_loss_on_decoded
            g_loss = self.config.adversarial_loss * g_loss_adv + self.ssim_weight * (1 - g_loss_enc_ssim) + \
                     self.config.encoder_loss * g_loss_enc + self.config.decoder_loss * g_loss_dec + \
                     g_loss_d * self.discriminator_message_weight

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item(),
            'encoded_ssim   ': g_loss_enc_ssim.item(),
            'discrim_message_real': d_loss_on_real.item(),
            'discrim_message_fake': d_loss_on_decoded.item(),
            'message_bce    ': g_loss_d.item(),
            'PSNR           ': 10 * torch.log10(4 / g_loss_enc).item(),
            'SSIM           ': g_loss_enc_ssim.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def to_stirng(self):
        return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))
