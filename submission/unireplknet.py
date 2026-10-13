"""UniRepLKNet-S classification implementation for Studio 2.

Adapted from the official AILab-CVC UniRepLKNet implementation:
https://github.com/AILab-CVC/UniRepLKNet
Original license: Apache-2.0. See LICENSES/Apache-2.0-UniRepLKNet.txt.

This local module contains the classification architecture needed by this
repository. It does not download source code. The ImageNet-1K pretrained
checkpoint is downloaded/cached only when training requests it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint as torch_checkpoint


def _to_2tuple(value):
    return value if isinstance(value, tuple) else (value, value)


class DropPath(nn.Module):
    """Per-sample stochastic depth. No parameters or buffers."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class GRNwithNHWC(nn.Module):
    """Global Response Normalization for NHWC activations."""
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * nx + 1) * x + self.beta
        return (self.gamma * nx + 1) * x


class NCHWtoNHWC(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class LayerNorm(nn.Module):
    """LayerNorm that supports channels-last and channels-first tensors."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last",
                 reshape_last_to_first=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if data_format not in ("channels_last", "channels_first"):
            raise NotImplementedError(f"Unsupported data_format={data_format}")
        self.normalized_shape = (normalized_shape,)
        self.reshape_last_to_first = reshape_last_to_first

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def get_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=None,
               dilation=1, groups=1, bias=False, attempt_use_lk_impl=False):
    # The optional iGEMM extension is deliberately not required for this task.
    kernel_size = _to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = _to_2tuple(padding)
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block; module names match official weights."""
    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        self.down = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.nonlinear(self.down(x))
        x = torch.sigmoid(self.up(x))
        return inputs * x.view(-1, self.input_channels, 1, 1)


class DilatedReparamBlock(nn.Module):
    """Large-kernel depthwise convolution represented by parallel branches."""
    def __init__(self, channels, kernel_size, deploy=False, use_sync_bn=False,
                 attempt_use_lk_impl=False):
        super().__init__()
        self.lk_origin = get_conv2d(
            channels, channels, kernel_size, stride=1, padding=kernel_size // 2,
            dilation=1, groups=channels, bias=deploy,
            attempt_use_lk_impl=attempt_use_lk_impl,
        )
        self.attempt_use_lk_impl = attempt_use_lk_impl
        # Lists and attribute names intentionally match the official checkpoint.
        branch_config = {
            17: ([5, 9, 3, 3, 3], [1, 2, 4, 5, 7]),
            15: ([5, 7, 3, 3, 3], [1, 2, 3, 5, 7]),
            13: ([5, 7, 3, 3, 3], [1, 2, 3, 4, 5]),
            11: ([5, 5, 3, 3, 3], [1, 2, 3, 4, 5]),
            9:  ([5, 5, 3, 3], [1, 2, 3, 4]),
            7:  ([5, 3, 3], [1, 2, 3]),
            5:  ([3, 3], [1, 2]),
        }
        if kernel_size not in branch_config:
            raise ValueError("DilatedReparamBlock requires kernel_size in {5,7,9,11,13,15,17}")
        self.kernel_sizes, self.dilates = branch_config[kernel_size]
        if not deploy:
            self.origin_bn = nn.SyncBatchNorm(channels) if use_sync_bn else nn.BatchNorm2d(channels)
            for k, rate in zip(self.kernel_sizes, self.dilates):
                self.__setattr__(
                    f"dil_conv_k{k}_{rate}",
                    nn.Conv2d(
                        in_channels=channels, out_channels=channels,
                        kernel_size=k, stride=1,
                        padding=(rate * (k - 1) + 1) // 2,
                        dilation=rate, groups=channels, bias=False,
                    ),
                )
                self.__setattr__(
                    f"dil_bn_k{k}_{rate}",
                    nn.SyncBatchNorm(channels) if use_sync_bn else nn.BatchNorm2d(channels),
                )

    def forward(self, x):
        if not hasattr(self, "origin_bn"):
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, rate in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__(f"dil_conv_k{k}_{rate}")
            bn = self.__getattr__(f"dil_bn_k{k}_{rate}")
            out = out + bn(conv(x))
        return out


class UniRepLKNetBlock(nn.Module):
    def __init__(self, dim, kernel_size, drop_path=0.0,
                 layer_scale_init_value=1e-6, deploy=False,
                 attempt_use_lk_impl=False, with_cp=False,
                 use_sync_bn=False, ffn_factor=4):
        super().__init__()
        self.with_cp = with_cp
        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(
                dim, kernel_size, deploy=deploy, use_sync_bn=use_sync_bn,
                attempt_use_lk_impl=attempt_use_lk_impl,
            )
        else:
            if kernel_size not in (3, 5):
                raise ValueError(f"Unsupported kernel_size={kernel_size}")
            self.dwconv = get_conv2d(
                dim, dim, kernel_size=kernel_size, stride=1,
                padding=kernel_size // 2, dilation=1, groups=dim,
                bias=deploy, attempt_use_lk_impl=attempt_use_lk_impl,
            )
        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = nn.SyncBatchNorm(dim) if use_sync_bn else nn.BatchNorm2d(dim)
        self.se = SEBlock(dim, dim // 4)
        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(NCHWtoNHWC(), nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(nn.GELU(), GRNwithNHWC(ffn_dim, use_bias=not deploy))
        if deploy:
            self.pwconv2 = nn.Sequential(nn.Linear(ffn_dim, dim), NHWCtoNCHW())
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim, bias=False),
                NHWCtoNCHW(),
                nn.SyncBatchNorm(dim) if use_sync_bn else nn.BatchNorm2d(dim),
            )
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if (not deploy) and layer_scale_init_value is not None and layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def compute_residual(self, x):
        y = self.se(self.norm(self.dwconv(x)))
        y = self.pwconv2(self.act(self.pwconv1(y)))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1) * y
        return self.drop_path(y)

    def forward(self, inputs):
        def residual(x):
            return x + self.compute_residual(x)
        if self.with_cp and inputs.requires_grad:
            return torch_checkpoint.checkpoint(residual, inputs, use_reentrant=False)
        return residual(inputs)


DEFAULT_KERNELS_S = (
    (3, 3, 3),
    (13, 13, 13),
    (13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3,
     13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3),
    (13, 13, 13),
)
DEPTHS_S = (3, 3, 27, 3)


class UniRepLKNet(nn.Module):
    """Classification form of UniRepLKNet with official checkpoint key names."""
    def __init__(self, in_chans=3, num_classes=1000, depths=DEPTHS_S,
                 dims=(96, 192, 384, 768), drop_path_rate=0.0,
                 layer_scale_init_value=1e-6, head_init_scale=1.0,
                 kernel_sizes=None, deploy=False, with_cp=False,
                 attempt_use_lk_impl=False, use_sync_bn=False):
        super().__init__()
        depths = tuple(depths)
        if kernel_sizes is None:
            if depths != DEPTHS_S:
                raise ValueError("No default kernel sizes configured for these depths")
            kernel_sizes = DEFAULT_KERNELS_S
        for i in range(4):
            if len(kernel_sizes[i]) != depths[i]:
                raise ValueError("kernel sizes do not match depths")

        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1),
                LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
                nn.GELU(),
                nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        ])
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                LayerNorm(dims[i + 1], eps=1e-6, data_format="channels_first"),
            ))

        self.stages = nn.ModuleList()
        current = 0
        for stage in range(4):
            blocks = [
                UniRepLKNetBlock(
                    dim=dims[stage], kernel_size=kernel_sizes[stage][block_idx],
                    drop_path=dp_rates[current + block_idx],
                    layer_scale_init_value=layer_scale_init_value, deploy=deploy,
                    attempt_use_lk_impl=attempt_use_lk_impl, with_cp=with_cp,
                    use_sync_bn=use_sync_bn,
                )
                for block_idx in range(depths[stage])
            ]
            self.stages.append(nn.Sequential(*blocks))
            current += depths[stage]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.output_mode = "logits"
        self.apply(self._init_weights)
        with torch.no_grad():
            self.head.weight.mul_(head_init_scale)
            self.head.bias.mul_(head_init_scale)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        for stage_idx in range(4):
            x = self.downsample_layers[stage_idx](x)
            x = self.stages[stage_idx](x)
        x = self.norm(x.mean([-2, -1]))
        return self.head(x)


def unireplknet_s(num_classes=1000, in_1k_pretrained=False,
                  drop_path_rate=0.0, attempt_use_lk_impl=False,
                  with_cp=False, **kwargs):
    """Build UniRepLKNet-S; optionally load official ImageNet-1K weights."""
    model = UniRepLKNet(
        num_classes=num_classes,
        depths=DEPTHS_S,
        dims=(96, 192, 384, 768),
        drop_path_rate=drop_path_rate,
        attempt_use_lk_impl=attempt_use_lk_impl,
        with_cp=with_cp,
        **kwargs,
    )
    if in_1k_pretrained:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "Install huggingface_hub to fetch the official ImageNet-1K checkpoint."
            ) from exc
        checkpoint_path = hf_hub_download(
            repo_id="DingXiaoH/UniRepLKNet",
            filename="unireplknet_s_in1k_224_acc83.91.pth",
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        model.load_state_dict(checkpoint, strict=True)
    return model
