# -*- coding: utf-8 -*-
# @Time    : 2022/11/8
# @Author  : White Jiang
import math

import torch.nn as nn
import torch
import torch.nn.functional as F


def get_list(w, k):
    width_list = []
    for i in range(1, math.ceil(w / k) + 1):
        if i == math.ceil(w / k):
            width_list.append(w - k * (i - 1))
        else:
            width_list.append(w // k)
    print(width_list)


# SSAM
class FESM(nn.Module):
    def __init__(self, in_channel=1024, k=2):
        super(FESM, self).__init__()
        self.k = k
        self.stripconv = nn.Sequential(
            # in_channel out_channel kernel_size stride padding
            nn.Conv2d(in_channel, 1, 1, 1, 0),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
        )
        self.stripconv_h = nn.Sequential(
            # in_channel out_channel kernel_size stride padding
            nn.Conv2d(in_channel, 1, 1, 1, 0),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # # 注意力机制生成阈值
        self.threshold_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化，输出尺寸为 [B, C, 1, 1]
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1),  # 使用 1x1 卷积生成一个通道
            nn.Sigmoid()  # 将阈值限制在 [0, 1] 之间
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.zeros_(self.threshold_attention[1].weight)
        nn.init.zeros_(self.threshold_attention[1].bias)

    def forward(self, fm):
        # print(fm.shape)
        b, c, w, h = fm.shape
        # 宽度方向上划分 k b c w/k h
        fm_w = torch.split(fm, w // self.k, dim=2)
        fm_h = torch.split(fm, h // self.k, dim=3)  # 高度方向划分
        # k b 1 w/k h
        fmw_conv = map(self.stripconv, fm_w)
        fmh_conv = map(self.stripconv_h, fm_h)
        # k b 1 1 1
        fmw_pool = list(map(self.avgpool, fmw_conv))
        fmh_pool = list(map(self.avgpool, fmh_conv))
        # print(fmh_pool[0].size())
        # k b 1 1 1 -> b 1 k 1
        fmw_pool = torch.cat(fmw_pool, dim=2)
        fmh_pool = torch.cat(fmh_pool, dim=3)
        # # soft计算用于注意力权重
        fmw_softmax = torch.softmax(fmw_pool, dim=2)  # every parts has one score [B*C*K*1]
        fmw_softmax = torch.repeat_interleave(fmw_softmax, w // self.k, dim=2)

        # print('fmw_softmax', fmw_softmax)

        fmh_softmax = torch.softmax(fmh_pool, dim=3)
        fmh_softmax = torch.repeat_interleave(fmh_softmax, w // self.k, dim=3)
        # b c h w
        fm_final = fmw_softmax * fmh_softmax

        alpha = 0.5

        fms_boost = fm + alpha * (fm * fm_final)
        # print(fms_boost[0][0])
        beta = 0.5

        # ---------------------------------------------------------
        # Content-dependent adaptive threshold
        # T: [B, 1, 1, 1], constrained to (0, 1) by Sigmoid
        # ---------------------------------------------------------
        adaptive_threshold = self.threshold_attention(fm)
        # adaptive_threshold = torch.full(
        #     (fm.size(0), 1, 1, 1),
        #     0.5,
        #     device=fm.device,
        #     dtype=fm.dtype
        # )
        # Normalize the spatial attention map by its maximum response.
        # fm_final: [B, 1, H, W]
        fms_max = fm_final.amax(dim=(2, 3), keepdim=True)
        normalized_score = fm_final / (fms_max + 1e-12)

        # ---------------------------------------------------------
        # Straight-Through Estimator (STE)
        #
        # Forward:
        #   hard_gate = 1 for dominant regions (score >= T)
        #               0 otherwise
        #
        # Backward:
        #   gradients are propagated through soft_gate
        # ---------------------------------------------------------
        soft_gate = torch.sigmoid(
            normalized_score - adaptive_threshold
        )

        hard_gate = (
                normalized_score >= adaptive_threshold
        ).float()

        gate = (
                hard_gate.detach()
                - soft_gate.detach()
                + soft_gate
        )

        # Dominant regions: mask = beta = 0.5
        # Other regions:    mask = 1.0
        fms_softmax_suppress = (
                1.0 - (1.0 - beta) * gate
        )

        # Apply the suppression mask
        fms_suppress = fm * fms_softmax_suppress
        # print(fms_suppress[0][0])
        return fms_boost, fms_suppress