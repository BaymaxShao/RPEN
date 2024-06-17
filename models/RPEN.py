import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from efficientnet_pytorch import EfficientNet
from einops import rearrange
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torchvision.models as tvmodels
from collections import OrderedDict
from einops import rearrange
from timm.models import register_model
from timm.models.layers import trunc_normal_, DropPath
from einops.layers.torch import Rearrange
from .JFE import mdires34
from .SFE import sfe_tiny
from thop import profile
from torchstat import stat


class RPEN(nn.Module):
    def __init__(self, out_dim=6):
        super(RPEN, self).__init__()
        self.out_dim = out_dim
        self.sep_features = sfe_tiny()# The whole code will be released when the work is accepted

        self.joint_features = mdires34()# The whole code will be released when the work is accepted

        self.squeeze = nn.Conv2d(512, 384, 1)
        self.relu = nn.ReLU(inplace=False)
        self.decoder = PoseDecoder()

    def forward(self, img1, img2):
        img3 = torch.cat([img1, img2], dim=1)
        f1 = self.sep_features(img1)
        f2 = self.sep_features(img2)
        f3 = self.joint_features(img3)
        feat = torch.cat((f1, f2, f3), dim=1)

        out = self.relu(self.squeeze(feat))

        out = self.decoder(out)

        pose = 0.01 * out.view(-1, self.out_dim)

        return pose


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class PoseDecoder(nn.Module):
    # The whole code will be released when the work is accepted


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


if __name__ == '__main__':
    input1 = torch.randn((16, 3, 224, 224))
    input2 = torch.randn((16, 3, 224, 224))
    flow = torch.randn((16, 3, 224, 224))
    model = ab6()
    output = model(input1, input2)
    print(output)
