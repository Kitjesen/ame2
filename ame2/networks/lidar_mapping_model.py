"""Wider-context LiDAR ablation of the paper's shallow MappingNet.

This variant is not the paper architecture and is not selected by the PPO entry.
It adds an observation-mask channel and a second pooling scale to let missing
regions use measured heights farther than the shallow model's receptive field.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .ame2_model import MappingNet, MappingConfig


class LidarContextMappingNet(MappingNet):
    def __init__(self, cfg: MappingConfig, missing_height=-2.):
        super().__init__(cfg)
        self.missing_height = missing_height
        ch = cfg.cnn_channels
        self.enc[0] = nn.Conv2d(2, ch, 3, padding=1)
        self.context = nn.Sequential(
            nn.Conv2d(ch, 2 * ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2 * ch, 2 * ch, 3, padding=1), nn.ReLU(inplace=True))
        self.context_dec = nn.Sequential(
            nn.Conv2d(3 * ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x, observed=None):
        if observed is None:
            observed = x != self.missing_height
        skip = self.enc(torch.cat((x, observed.to(x.dtype)), 1))
        low = self.pool(skip)
        context = self.context(self.pool(low))
        context = F.interpolate(context, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.context_dec(torch.cat((low, context), 1))
        up = F.interpolate(low, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.dec(torch.cat((up, skip), 1))
        gate = torch.sigmoid(self.head_gate(feat))
        elevation = gate * self.head_elev(feat) + (1. - gate) * x
        log_var = self.head_unc(feat).clamp_min(math.log(self.MIN_VARIANCE))
        return elevation, log_var
