# file: model_relu.py
"""
ReLU variant of DeepPhysEnhanced.
Only change: replace all Tanh activations with ReLU, preserve structure and behavior elsewhere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5


class DeepPhysEnhanced(nn.Module):
    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3,
                 dropout_rate1=0.25, dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, img_size=36):
        super(DeepPhysEnhanced, self).__init__()

        self.in_channels = in_channels
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2

        # Motion branch
        self.motion_conv1 = nn.Conv2d(in_channels, nb_filters1, kernel_size=kernel_size, padding=1)
        self.motion_conv2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size=kernel_size, padding=1)
        self.motion_conv3 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size=kernel_size, padding=1)
        self.motion_conv4 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size=kernel_size, padding=1)

        # Appearance branch
        self.appearance_conv1 = nn.Conv2d(in_channels, nb_filters1, kernel_size=kernel_size, padding=1)
        self.appearance_conv2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size=kernel_size, padding=1)
        self.appearance_conv3 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size=kernel_size, padding=1)
        self.appearance_conv4 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size=kernel_size, padding=1)

        self.appearance_att_conv1 = nn.Conv2d(nb_filters1, 1, kernel_size=1)
        self.attn_mask_1 = Attention_mask()
        self.appearance_att_conv2 = nn.Conv2d(nb_filters2, 1, kernel_size=1)
        self.attn_mask_2 = Attention_mask()

        # POS+Green branch
        self.posgreen_conv1 = nn.Conv2d(2, nb_filters1, kernel_size=kernel_size, padding=1)
        self.posgreen_conv2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size=kernel_size, padding=1)
        self.posgreen_conv3 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size=kernel_size, padding=1)
        self.posgreen_conv4 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size=kernel_size, padding=1)

        # Pooling and dropout
        self.avg_pooling = nn.AvgPool2d(pool_size)
        self.dropout_1 = nn.Dropout(dropout_rate1)
        self.dropout_2 = nn.Dropout(dropout_rate1)
        self.dropout_3 = nn.Dropout(dropout_rate1)
        self.dropout_posgreen = nn.Dropout(dropout_rate1)
        self.dropout_4 = nn.Dropout(dropout_rate2)

        # Final FC layers
        self.flat_dim = nb_filters2 * (img_size // 4) * (img_size // 4) * 2
        self.final_dense_1 = nn.Linear(self.flat_dim, nb_dense)
        self.final_dense_2 = nn.Linear(nb_dense, 1)

    def forward(self, diff_input, raw_input, posgreen_input):
        # Vérifie et squeeze si nécessaire
        if posgreen_input.ndim == 5 and posgreen_input.shape[2] == 1:
            posgreen_input = posgreen_input.squeeze(2)

        # Motion branch
        d1 = F.relu(self.motion_conv1(diff_input))
        d2 = F.relu(self.motion_conv2(d1))

        # Appearance branch
        r1 = F.relu(self.appearance_conv1(raw_input))
        r2 = F.relu(self.appearance_conv2(r1))
        g1 = torch.sigmoid(self.appearance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling(gated1)
        d4 = self.dropout_1(d3)
        r3 = self.avg_pooling(r2)
        r4 = self.dropout_2(r3)

        d5 = F.relu(self.motion_conv3(d4))
        d6 = F.relu(self.motion_conv4(d5))
        r5 = F.relu(self.appearance_conv3(r4))
        r6 = F.relu(self.appearance_conv4(r5))

        g2 = torch.sigmoid(self.appearance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2
        d7 = self.avg_pooling(gated2)
        d8 = self.dropout_3(d7)

        # POS + Green branch
        b1 = F.relu(self.posgreen_conv1(posgreen_input))
        b2 = F.relu(self.posgreen_conv2(b1))
        b3 = F.relu(self.posgreen_conv3(b2))
        b4 = F.relu(self.posgreen_conv4(b3))
        b4 = self.avg_pooling(b4)
        b4 = self.dropout_posgreen(b4)

        # Harmoniser tailles spatiales avant concat
        if d8.shape[2:] != b4.shape[2:]:
            b4 = F.adaptive_avg_pool2d(b4, d8.shape[2:])

        merged = torch.cat([d8, b4], dim=1)
        merged_flat = merged.view(merged.size(0), -1)
        d10 = F.relu(self.final_dense_1(merged_flat))
        d11 = self.dropout_4(d10)
        return self.final_dense_2(d11)
