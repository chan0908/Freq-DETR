# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from typing import Tuple, Optional, List, Sequence

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'ResNetLayer','CSP_FreqSpatial','AGS_FPN')


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

######################################## FECM start ########################################

class DeformableConvPack(nn.Module):

    def __init__(self, in_channels, out_channels, k=3, stride=1, dilation=1):
        super().__init__()
        self.k = k
        self.stride = stride
        self.dilation = dilation
        self.pad = autopad(k, dilation=dilation)

        self.use_dcn = hasattr(torchvision.ops, 'deform_conv2d')

        if self.use_dcn:
            self.offset_conv = nn.Conv2d(
                in_channels, 2 * k * k, kernel_size=k,
                padding=self.pad, stride=stride, dilation=dilation
            )
            self.weight = nn.Parameter(torch.empty(out_channels, in_channels, k, k))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.fallback = Conv(in_channels, out_channels, k=k, s=stride, d=dilation)

    def forward(self, x):
        if self.use_dcn:
            offset = self.offset_conv(x)
            y = deform_conv2d(
                input=x, offset=offset, weight=self.weight, bias=self.bias,
                stride=self.stride, padding=self.pad, dilation=self.dilation
            )
            return y
        else:
            return self.fallback(x)


class ScharrConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Scharr滤波器核
        kx = np.array([[3, 0, -3],
                       [10, 0, -10],
                       [3, 0, -3]], dtype=np.float32)
        ky = np.array([[3, 10, 3],
                       [0, 0, 0],
                       [-3, -10, -3]], dtype=np.float32)

        self.convx = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.convy = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

        with torch.no_grad():
            wx = torch.from_numpy(kx)[None, None, ...].repeat(channels, 1, 1, 1)
            wy = torch.from_numpy(ky)[None, None, ...].repeat(channels, 1, 1, 1)
            self.convx.weight.copy_(wx)
            self.convy.weight.copy_(wy)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        gx = self.convx(x)
        gy = self.convy(x)
        return 0.5 * gx + 0.5 * gy


class DynamicFreqConv(nn.Module):

    def __init__(self, channels: int, n_kernels: int = 4, k: int = 3):
        super().__init__()
        self.nk = n_kernels
        pad = k // 2

        self.dw_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, k, padding=pad, groups=channels, bias=False)
            for _ in range(n_kernels)
        ])
        self.dw_bn = nn.BatchNorm2d(channels)
        self.weight_gen = nn.Conv2d(channels, n_kernels, kernel_size=1)
        self.pw = Conv(channels, channels, k=1)

    def forward(self, x):
        ys = [conv(x) for conv in self.dw_convs]
        ystk = torch.stack(ys, dim=1)  # [B, n_kernels, C, H, W]
        w = F.softmax(self.weight_gen(x), dim=1).unsqueeze(2)  # [B, n_kernels, 1, H, W]
        y = (ystk * w).sum(dim=1)
        y = self.dw_bn(y)
        return self.pw(y)


class HighFreqFDM(nn.Module):

    def __init__(self, channels: int, num_blocks: int = 4, softshrink=0.0, activation="relu"):
        super().__init__()
        C = channels
        self.Cb = num_blocks
        self.Cd = C // self.Cb

        act = nn.ReLU if activation.lower() == "relu" else nn.GELU

        # 复数线性变换层 (实部和虚部)
        self.lh_1_0 = nn.Linear(self.Cd, self.Cd)
        self.lh_1_1 = nn.Linear(self.Cd, self.Cd)
        self.lh_2_0 = nn.Linear(self.Cd, self.Cd)
        self.lh_2_1 = nn.Linear(self.Cd, self.Cd)

        self.act = act()
        self.softshrink = nn.Softshrink(lambd=softshrink) if softshrink and softshrink > 0 else None

    def forward(self, yh_j):
        n, c, o, h, w, ri = yh_j.shape
        assert c == self.Cb * self.Cd and ri == 2 and o == 6, \
            f"Expected shape [N, {self.Cb * self.Cd}, 6, H, W, 2], got {yh_j.shape}"

        t = yh_j.permute(5, 0, 2, 3, 4, 1).contiguous()
        t = t.view(2, n, o, h, w, self.Cb, self.Cd)

        x_real = t[0]  # 实部
        x_imag = t[1]  # 虚部

        def flat(x):
            return x.contiguous().view(-1, self.Cd)

        def unflat(x, ref):
            return x.view(ref.shape)

        # 展平处理
        xr = flat(x_real)
        xi = flat(x_imag)

        xr1 = self.lh_1_0(xr) - self.lh_1_1(xi)
        xi1 = self.lh_1_1(xr) + self.lh_1_0(xi)
        xr1 = self.act(xr1)
        xi1 = self.act(xi1)
        xr2 = self.lh_2_0(xr1) - self.lh_2_1(xi1)
        xi2 = self.lh_2_1(xr1) + self.lh_2_0(xi1)

        xr2 = unflat(xr2, x_real)
        xi2 = unflat(xi2, x_imag)

        y = torch.stack([xr2, xi2], dim=0)

        if self.softshrink is not None:
            y = self.softshrink(y)

        y = y.view(2, n, o, h, w, c)
        y = y.permute(1, 5, 2, 3, 4, 0).contiguous()
        return y


class DTCWTBranch(nn.Module):

    def __init__(self, in_channels: int, J: int = 1,
                 biort: str = 'near_sym_b', qshift: str = 'qshift_b',
                 n_kernels_dfc: int = 4,
                 fdm_blocks: int = 4,
                 fdm_softshrink=0.0):
        super().__init__()
        if not DTCWT_AVAILABLE:
            raise ImportError("pytorch_wavelets is required for DTCWTBranch")

        self.J = J
        self.xfm = DTCWTForward(J=J, biort=biort, qshift=qshift)
        self.ifm = DTCWTInverse(biort=biort, qshift=qshift)

        self.dfc = DynamicFreqConv(in_channels, n_kernels=n_kernels_dfc, k=3)
        self.hf_fdm = nn.ModuleList([
            HighFreqFDM(in_channels, num_blocks=fdm_blocks,
                        softshrink=fdm_softshrink, activation="relu")
            for _ in range(J)
        ])

        self.alpha_l = nn.Parameter(torch.tensor(1.0))
        self.alpha_h = nn.Parameter(torch.tensor(1.0))
        self.post = Conv(in_channels, in_channels, k=3)

        self.export_hf = True
        self.last_hf = None

    def forward(self, x, return_hf=False):
        yl, yh = self.xfm(x)

        # 处理低频分量
        yl = self.dfc(yl) * self.alpha_l

        # 处理各尺度的高频分量
        hf_processed = []
        for j in range(self.J):
            yj = self.hf_fdm[j](yh[j]) * self.alpha_h
            hf_processed.append(yj)

        # 生成高频特征图
        if return_hf or self.export_hf:
            up_list = []
            for j in range(self.J):
                yj = hf_processed[j]
                mag = yj[..., 0].abs() + yj[..., 1].abs()
                mag = mag.mean(dim=2)
                up_list.append(F.interpolate(mag, size=yl.shape[-2:], mode='bilinear', align_corners=False))
            hf_skip = torch.stack(up_list, dim=0).sum(0)
        else:
            hf_skip = None

        y = self.ifm((yl, tuple(hf_processed)))
        y = self.post(y)

        self.last_hf = hf_skip
        if return_hf:
            return y, hf_skip
        return y


class SpatialBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scharr = ScharrConv(channels)
        self.W1 = Conv(channels, channels, k=3, g=channels)
        self.W2 = Conv(channels, channels, k=3)
        self.dilated = nn.Sequential(
            Conv(channels, channels, k=3, d=2),
            Conv(channels, channels, k=3, d=4),
        )

    def forward(self, F_sp):
        G = self.scharr(F_sp)
        part1 = F.silu(self.W1(G))
        part2 = F.silu(self.W2(F_sp + G))
        scharr_out = part1 + part2
        dil_out = self.dilated(F_sp)
        return scharr_out + dil_out


class CSP_FreqSpatial(nn.Module):
    def __init__(self, in_channels: int,
                 dtcwt_levels: int = 1, dfc_kernels: int = 4,
                 fdm_blocks: str | int = "auto", fdm_softshrink: float = 0.0):
        super().__init__()
        self.in_c = in_channels

        self.pre_pw = Conv(in_channels, in_channels * 2, k=1)

        # 两个分支
        self.spatial = SpatialBranch(in_channels)
        self.freq = DTCWTBranch(in_channels, J=dtcwt_levels,
                                n_kernels_dfc=dfc_kernels,
                                fdm_blocks=fdm_blocks,
                                fdm_softshrink=fdm_softshrink)

        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.align_sp = Conv(in_channels, in_channels, k=1)
        self.align_fr = Conv(in_channels, in_channels, k=1)

        self.dcn1 = DeformableConvPack(in_channels * 2, in_channels, k=3)
        self.dcn2 = DeformableConvPack(in_channels * 2, in_channels, k=3)

        self.post_pw = Conv(in_channels, in_channels, k=1)

    def forward(self, x, return_hf=False):
        t = self.pre_pw(x)
        F_sp, F_fr = torch.chunk(t, 2, dim=1)

        # 两个分支处理
        X_spatial = self.spatial(F_sp)
        if return_hf:
            X_freq, hf = self.freq(F_fr, return_hf=True)
        else:
            X_freq = self.freq(F_fr)
            hf = None

        # 特征对齐
        X_spatial = self.align_sp(X_spatial)
        X_freq = self.align_fr(X_freq)

        # 双路径融合
        X1 = torch.cat([X_spatial * self.alpha, X_freq], dim=1)
        X2 = torch.cat([X_spatial, X_freq * self.alpha], dim=1)
        Y1 = self.dcn1(X1)
        Y2 = self.dcn2(X2)

        y = self.post_pw(Y1 + Y2)
        if return_hf:
            return y, hf
        return y


######################################## FECM end ########################################


#######################################  AGS-FPN start ###################################
class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=0, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv2d(nn.Module):
    def __init__(self, c, k: Tuple[int, int], s=1, p: Tuple[int, int] = (0, 0), d=1, act=True):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, s, p, groups=c, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))


class SACA(nn.Module):

    def __init__(self, ch: int, kb: int = 11, pool_k: int = 7, use_ds_pointwise: bool = False):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_k, 1, pool_k // 2)
        self.reduce = ConvBNAct(ch, ch, k=1, act=True)
        self.v_dw = DWConv2d(ch, k=(1, kb), p=(0, kb // 2), act=True)
        self.h_dw = DWConv2d(ch, k=(kb, 1), p=(kb // 2, 0), act=True)
        self.out_1x1 = nn.Conv2d(ch, ch, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.use_ds_pointwise = use_ds_pointwise
        if use_ds_pointwise:
            self.pw = ConvBNAct(ch, ch, k=1, act=True)

    def forward(self, x, return_weight_only: bool = False):
        a = self.pool(x)
        a = self.reduce(a)
        a = self.v_dw(a)
        a = self.h_dw(a)
        if self.use_ds_pointwise:
            a = self.pw(a)
        A_DA = self.sigmoid(self.out_1x1(a))
        if return_weight_only:
            return A_DA
        refined = x * A_DA + x
        return refined, A_DA


class DTBlock(nn.Module):

    def __init__(self, ch: int = 256, dilation: int = 2, use_hf_gate: bool = True):
        super().__init__()
        self.pre = ConvBNAct(ch, ch, k=1, act=True)
        self.dilated = ConvBNAct(ch, ch, k=3, p=0, d=dilation, act=True)
        self.use_hf_gate = use_hf_gate
        if use_hf_gate:
            self.hf_proj = nn.Conv2d(ch, ch, kernel_size=1, bias=True)
            self.hf_sig = nn.Sigmoid()

    def forward(self, f_high: torch.Tensor, hf: Optional[torch.Tensor] = None):
        y0 = self.pre(f_high)
        y = self.dilated(y0)
        target_size = f_high.shape[-2:]
        if y.shape[-2:] != target_size:
            y = F.interpolate(y, size=target_size, mode="bilinear", align_corners=False)
        if self.use_hf_gate and (hf is not None):
            assert hf.shape[1] == f_high.shape[1], \
                f"hf channels ({hf.shape[1]}) must equal FPN channels ({f_high.shape[1]}). " \
                f"use 1x1 "
            hf_up = F.interpolate(hf, size=target_size, mode="bilinear", align_corners=False)
            gate = self.hf_sig(self.hf_proj(hf_up))
            y = y * (1.0 + gate)

        return y + f_high


class AGS_FPN(nn.Module):
    def __init__(self,
                 in_channels: Sequence[int],
                 out_channels: int = 256,
                 kb: int = 11,
                 pool_k: int = 7,
                 use_hf_gate: bool = True):
        super().__init__()
        self.num_levels = len(in_channels)
        self.out_c = out_channels

        self.lateral = nn.ModuleList([ConvBNAct(c, out_channels, k=1, act=True) for c in in_channels])
        self.saca = SACA(out_channels, kb=kb, pool_k=pool_k, use_ds_pointwise=True)

        self.dtblocks = nn.ModuleList([
            DTBlock(ch=out_channels, dilation=2, use_hf_gate=use_hf_gate)
            for _ in range(self.num_levels - 1)
        ])
        self.smooth = nn.ModuleList([
            ConvBNAct(out_channels, out_channels, k=3, p=1, act=True)
            for _ in range(self.num_levels)
        ])

    def forward(self,
                feats: List[torch.Tensor],
                hf_list: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:

        if hf_list is not None:
            assert len(hf_list) == self.num_levels
        lat = [self.lateral[i](feats[i]) for i in range(self.num_levels)]

        outs = [None] * self.num_levels
        top_idx = self.num_levels - 1

        high = lat[top_idx]
        outs[top_idx] = self.smooth[top_idx](high)
        for i in reversed(range(self.num_levels - 1)):
            target_size = lat[i].shape[-2:]
            high_up = F.interpolate(high, size=target_size, mode="bilinear", align_corners=False)
            hf = hf_list[i] if (hf_list is not None) else None
            high_aligned = self.dtblocks[i](high_up, hf=hf)
            A_DA = self.saca(high_aligned, return_weight_only=True)
            filtered = lat[i] * A_DA
            fused = filtered + high_aligned
            outs[i] = self.smooth[i](fused)
            high = outs[i]

        return outs
#######################################  AGS-FPN end #####################################










class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(Conv(c1, c2, k=7, s=2, p=3, act=True),
                                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)
