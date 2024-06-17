from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from einops import rearrange


class StemLayer(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=96,
                 act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = norm_layer(out_channels // 2)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(out_channels // 2,
                               out_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels=96, out_channels=198, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class GatedCNNBlock(nn.Module):
    # The whole code will be released when the work is accepted


class SFE(nn.Module):
    def __init__(self, in_chans=3,
                 depths=[3, 3],
                 dims=[96, 192],
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 conv_ratio=1.0,
                 kernel_size=7,
                 drop_path_rate=0.,
                 output_norm=partial(nn.LayerNorm, eps=1e-6),
                 **kwargs,
                 ):
        super().__init__()

        if not isinstance(depths, (list, tuple)):
            depths = [depths]
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(num_stage)]
        )

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[GatedCNNBlock(dim=dims[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                kernel_size=kernel_size,
                conv_ratio=conv_ratio,
                drop_path=dp_rates[cur + j],
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = output_norm(dims[-1])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x):
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return rearrange(x, 'b h w c -> b c h w')

    def forward(self, x):
        x = self.forward_features(x)
        return x

@register_model
def sfe_tiny(pretrained=False, **kwargs):
    model = SFE(
        depths=[3, 3],
        dims=[96, 192],
        **kwargs)
    return model

