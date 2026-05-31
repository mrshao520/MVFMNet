import math
import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY


from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def get_local_weights(residual, ksize, padding):
    pad = padding
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode="reflect")
    unfolded_residual = residual_pad.unfold(2, ksize, 3).unfold(3, ksize, 3)
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    return pixel_level_weight


def adaptive_get_local_weights(residual, out_size):
    """计算给定残差的局部方差"""
    _, _, h, w = residual.shape
    ph, pw = out_size
    stride_h = h // ph
    stride_w = w // pw
    kernel_h = h - (ph - 1) * stride_h
    kernel_w = w - (pw - 1) * stride_w

    unfolded_residual = residual.unfold(2, kernel_h, stride_h).unfold(3, kernel_w, stride_w)
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)
    return pixel_level_weight


class BSConvU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


class PAConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        padding=2,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        partil=0.5, # 0.25 0.5 0.75 1
        down_scale=8,
    ):
        super().__init__()

        self.down_scale = down_scale
        self.remaining_channels = int(in_channels * partil)
        self.other_channels = in_channels - self.remaining_channels
        
        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # partial depth-wise
        self.pdw = nn.Conv2d(
            in_channels=self.remaining_channels,
            out_channels=self.remaining_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=self.remaining_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.gelu = nn.GELU()

    def forward(self, fea):
        _, _, h, w = fea.shape
        # channel split
        fea1, fea2 = torch.split(fea, [self.remaining_channels, self.other_channels], dim=1)

        # Attention
        fea1_s = self.pdw(F.adaptive_max_pool2d(fea1, (h // self.down_scale, w // self.down_scale)))
        fea1 = fea1 * F.interpolate(self.gelu(fea1_s), size=(h, w), mode="nearest")

        # channel shuffle
        fea = torch.cat((fea1, fea2), 1)
        fea = self.pw(fea)
        return fea


class MASA(nn.Module):
    def __init__(self, num_feat=50, conv=BSConvU):
        super().__init__()

        f = num_feat // 4

        self.in_proj = nn.Conv2d(num_feat, f, 1)

        self.conv_f = nn.Conv2d(f, f, 1)

        self.p1 = PAConv(f, f, kernel_size=1, padding=0, down_scale=8)
        self.p2 = PAConv(f, f, kernel_size=3, padding=1, down_scale=4)
        self.p3 = PAConv(f, f, kernel_size=5, padding=2, down_scale=1)

        self.conv1 = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, padding=1)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)

        self.out_proj = nn.Conv2d(f, num_feat, 1)

        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c_input = self.in_proj(input)  # channel squeeze

        p1 = self.conv1(self.GELU(self.p1(c_input)))
        p2 = self.conv2(self.GELU(self.p2(c_input)))
        p3 = self.conv3(self.GELU(self.p3(c_input)))

        p4 = self.conv_f(c_input)

        out = self.out_proj((p1 + p2 + p3 + p4))

        out = self.sigmoid(out)

        return input * out + input


class APFDB(nn.Module):

    def __init__(self, in_channels, out_channels, atten_channels=None):
        super().__init__()

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        if atten_channels is None:
            self.atten_channels = in_channels
        else:
            self.atten_channels = atten_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = PAConv(in_channels, self.rc, kernel_size=1, padding=0, down_scale=8)

        self.c2_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c2_r = PAConv(in_channels, self.rc, kernel_size=3, padding=1, down_scale=4)

        self.c3_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c3_r = PAConv(in_channels, self.rc, kernel_size=5, padding=2, down_scale=1)

        self.c4 = BSConvU(self.rc, self.dc, kernel_size=3, padding=1)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, self.atten_channels, 1, 1, 0)

        self.sa = MASA(self.atten_channels)

        self.pixel_norm = nn.LayerNorm(out_channels)  # channel-wise
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.c1_r(input)
        r_c1 = self.act(r_c1)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c2)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c3)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)

        out = self.sa(out)

        out = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        out = self.pixel_norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return out + input


def UpsampleOneStep(in_channels, out_channels, upscale_factor=4):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class Upsampler_rep(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels * 2, 1)
        self.conv3x3 = nn.Conv2d(in_channels * 2, out_channels * (upscale_factor**2), 3)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        v1 = F.conv2d(x, self.conv1x1.weight, self.conv1x1.bias, padding=0)
        v1 = F.pad(v1, (1, 1, 1, 1), "constant", 0)
        b0_pad = self.conv1x1.bias.view(1, -1, 1, 1)
        v1[:, :, 0:1, :] = b0_pad
        v1[:, :, -1:, :] = b0_pad
        v1[:, :, :, 0:1] = b0_pad
        v1[:, :, :, -1:] = b0_pad
        v2 = F.conv2d(v1, self.conv3x3.weight, self.conv3x3.bias, padding=0)
        out = self.conv1(x) + self.conv3(x) + v2
        return self.pixel_shuffle(out)


@ARCH_REGISTRY.register()
class PAFAN(nn.Module):
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=36,
        num_atten=36,
        num_block=8,
        upscale=4,
        num_in=4,
        upsampler="pixelshuffledirect",
        rgb_mean=(0.4488, 0.4371, 0.4040),
    ):
        super().__init__()
        self.num_in = num_in
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.fea_conv = BSConvU(num_in_ch * num_in, num_feat, kernel_size=3, padding=1)

        self.B1 = APFDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten)
        self.B2 = APFDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten)
        self.B3 = APFDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten)
        self.B4 = APFDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten)
        self.B5 = APFDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten)
        self.B6 = APFDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten)
        self.B7 = APFDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten)
        self.B8 = APFDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1, 1, 0)
        self.GELU = nn.GELU()

        self.c2 = BSConvU(num_feat, num_feat, kernel_size=3, padding=1)

        if upsampler == "pixelshuffledirect":
            self.upsampler = UpsampleOneStep(num_feat, num_out_ch, upscale_factor=upscale)
        elif upsampler == "pixelshuffle_rep":
            self.upsampler = Upsampler_rep(num_feat, num_out_ch, upscale_factor=upscale)
        else:
            raise NotImplementedError("Check the Upsampler. None or not support yet.")

    def forward(self, input):
        self.mean = self.mean.type_as(input)
        input = input - self.mean
        input = torch.cat([input] * self.num_in, dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)
        
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea
        output = self.upsampler(out_lr) + self.mean

        return output


if __name__ == "__main__":
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    # x = torch.randn(1, 3, 640, 360) # x2
    # x = torch.randn(1, 3, 427, 240)  # x3
    x = torch.randn(1, 3, 320, 180)  # x4
    # x = torch.randn(1, 3, 256, 256)

    model = PAFAN(num_feat=36, num_atten=36, num_block=8)
    print(f"params: {sum(map(lambda x: x.numel(), model.parameters()))}")
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)
