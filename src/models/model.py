# model.py
# EfficientNetB0 + FastKAN architecture
# Dùng để load best_EfficientNetB0_FastKAN.pth trong app realtime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FastKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size=8, sigma=0.5):
        super().__init__()

        self.sigma = sigma

        grid = torch.linspace(-1, 1, grid_size)
        self.register_buffer("grid", grid)

        self.base = nn.Linear(input_dim, output_dim)

        self.spline_weight = nn.Parameter(
            torch.randn(output_dim, input_dim, grid_size) * 0.01
        )

    def forward(self, x):
        base_out = self.base(x)

        x_expanded = x.unsqueeze(-1)

        rbf = torch.exp(
            -((x_expanded - self.grid) ** 2) / (2 * self.sigma ** 2)
        )

        spline_out = torch.einsum(
            "big,oig->bo",
            rbf,
            self.spline_weight
        )

        return base_out + spline_out


class FastKAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=8, sigma=0.5, dropout=0.5):
        super().__init__()

        self.layers = nn.ModuleList()

        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(
                FastKANLayer(
                    input_dim=in_dim,
                    output_dim=out_dim,
                    grid_size=grid_size,
                    sigma=sigma
                )
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.silu(x)
            x = self.dropout(x)

        return self.layers[-1](x)


class EfficientNetB0_FastKAN(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super().__init__()

        # Khi chạy app, pretrained=False vì trọng số đã nằm trong file .pth
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None

        backbone = models.efficientnet_b0(weights=weights)

        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # EfficientNet-B0 output feature dim = 1280
        self.feature_dim = 1280

        self.fastkan = FastKAN(
            layers_hidden=[self.feature_dim, 512, 128, num_classes],
            grid_size=8,
            sigma=0.5,
            dropout=0.5
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fastkan(x)

        return x