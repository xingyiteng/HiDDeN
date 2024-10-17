import torch.nn as nn
# from kornia.color.adjust import AdjustHue,AdjustSaturation,AdjustContrast,AdjustBrightness,AdjustGamma

from kornia.enhance import AdjustContrast


class Adjust_contrast(nn.Module):
    def __init__(self,factor):
        super(Adjust_contrast, self).__init__()
        self.factor=factor

    def forward(self, noised_and_cover):
        encoded=((noised_and_cover[0]).clone())
        encoded=AdjustContrast(contrast_factor=self.factor)(encoded)
        noised_and_cover[0]=(encoded)
        return noised_and_cover