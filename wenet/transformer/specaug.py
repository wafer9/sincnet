#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright to espnet
https://github.com/espnet/espnet/blob/master/espnet2/asr/specaug/specaug.py

Modified into spec aug with masking by ratio by Huahuan Zheng (zhh20@mails.tsinghua.edu.cn) in 2021.
"""

from typing import Union
from typing import Sequence
import torch
import numpy as np


class MaskFreq(torch.nn.Module):
    def __init__(
        self,
        mask_width_range: int = 10,
        num_mask: int = 2
    ):
        super().__init__()
        self.mask_width_range = mask_width_range
        self.num_mask = num_mask

    def forward(self, spec: torch.Tensor):
        """Apply mask along the freq direction.
        Args:
            spec: (batch, length, freq) or (batch, channel, length, freq)
        """


        if self.num_mask == 0:
            return spec

        B, _, D = spec.shape
        # mask_length: (B, num_mask, 1)
        mask_length = torch.randint(
            1,
            self.mask_width_range,
            (B, self.num_mask),
            device=spec.device,
        ).unsqueeze(2)

        # mask_pos: (B, num_mask, 1)
        mask_pos = torch.randint(
            0, max(1, D - mask_length.max()), (B, self.num_mask), device=spec.device
        ).unsqueeze(2)

        # aran: (1, 1, D)
        aran = torch.arange(D, device=spec.device)[None, None, :]
        # mask: (Batch, num_mask, D)
        mask = (mask_pos <= aran) * (aran < (mask_pos + mask_length))
        # Multiply masks: (Batch, num_mask, D) -> (Batch, D)
        mask = mask.any(dim=1)

        # mask: (Batch, 1, Freq)
        mask = mask.unsqueeze(1)

        spec = spec.masked_fill(mask, 0.0)
        return spec


class MaskTime(torch.nn.Module):
    def __init__(
        self,
        mask_width_range: int = 50,
        num_mask: int = 2
    ):

        super().__init__()
        self.mask_width_range = mask_width_range
        self.num_mask = num_mask

    def forward(self, spec: torch.Tensor):
        """Apply mask along time direction.
        Args:
            spec: (batch, length, freq) or (batch, channel, length, freq)
            spec_lengths: (length)
        """
        if self.num_mask == 0:
            return spec

        B, L, _ = spec.shape
        mask_len = torch.randint(
            low=1,
            high=self.mask_width_range,
            size=(B, self.num_mask),
            device=spec.device
        ).unsqueeze(2)
        mask_pos = torch.randint(
            0, max(1, L - mask_len.max()), (B, self.num_mask), device=spec.device
        ).unsqueeze(2)
        aran = torch.arange(L, device=spec.device)[None, None, :]
        mask = (mask_pos <= aran) * (aran < (mask_pos + mask_len))
        mask = mask.any(dim=1).unsqueeze(2)

        spec = spec.masked_fill(mask, 0.0)
        return spec


class SpecAug(torch.nn.Module):
    """Implementation of SpecAug.

    Reference:
        Daniel S. Park et al.
        "SpecAugment: A Simple Data
         Augmentation Method for Automatic Speech Recognition"
    """

    def __init__(
        self,
        apply_freq_mask: bool = True,
        freq_mask_width_range: int = 10,
        num_freq_mask: int = 2,
        apply_time_mask: bool = True,
        time_mask_width_range: int = 50,
        num_time_mask: int = 2,
    ):
        super().__init__()
        self.apply_freq_mask = apply_freq_mask
        self.apply_time_mask = apply_time_mask

        self.freq_mask = MaskFreq(
            mask_width_range=freq_mask_width_range,
            num_mask=num_freq_mask,
        )

        self.time_mask = MaskTime(
            mask_width_range=time_mask_width_range,
            num_mask=num_time_mask,
        )

    def forward(self, x):
        if self.apply_time_mask:
            x = self.time_mask(x)
        if self.apply_freq_mask:
            x = self.freq_mask(x)
        return x


if __name__ == '__main__':
    sp = SpecAug()
    x = torch.randn(2, 100, 80)
    y = sp(x)
    print(y.numpy().tolist())
