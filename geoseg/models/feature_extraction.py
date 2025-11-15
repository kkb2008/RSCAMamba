import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F

import torch.distributed as dist
from torch.autograd.function import Function
from torchvision.ops import DeformConv2d


class DeformableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=(0, 1), dilation=1, groups=1,
                 deformable_groups=1):
        super(DeformableConv, self).__init__()
        self.offset = nn.Conv2d(in_channels, deformable_groups * 2 * kernel_size[0] * kernel_size[1],
                                kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=False)

    def forward(self, x):
        offset = self.offset(x)

        offset[:, 1, :, :] = 0
        offset[:, 4, :, :] = 0

        x = self.deform_conv(x, offset)
        return x


class RDeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):

        super(RDeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        temp = x

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)

        q_lt = p.detach().floor()

        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        if self.modulation:  # m: (b,N,h,w)
            m = m.contiguous().permute(0, 2, 3, 1)  # (b,h,w,N)
            m = m.unsqueeze(dim=1)  # (b,1,h,w,N)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)  # (b,c,h,w,N)
            x_offset *= m 

        x_offset = x_offset.permute(0, 1, 4, 2, 3)
        x_offset[:, :, 4, :, :] = temp
        x_offset = x_offset.permute(0, 1, 3, 4, 2)

        # x_offset.shape = (b, c, h * ks, w * ks)
        x_offset = self._reshape_x_offset(x_offset, ks)
        # out.shape = (b, c, h, w)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            indexing="ij"
        )
        """
            p_n_x:
            tensor([[-1,  0,  1],
                    [-1,  0,  1],
                    [-1,  0,  1]])
            p_n_y:
            tensor([[-1, -1, -1],
                    [ 0,  0,  0],
                    [ 1,  1,  1]])
        """
        # p_n.shape(2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # self.stride = 1
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
            indexing="ij"
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        # q.shape = # (b, h, w, 2N)
        b, h, w, _ = q.size()
        # x: (b, c, h, w)
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):

        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)

        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


class SpatialFuse(nn.Module):
    def __init__(self, channel, kernel_size=3, padding=1, stride=1, bias=None):
        super(SpatialFuse, self).__init__()
        self.rDcn = RDeformConv2d(channel, channel, kernel_size, padding, stride, bias)

        self.deform_conv_1x3 = DeformableConv(channel, channel, kernel_size=(1, 3), padding=(0, 1))
        self.deform_conv_3x1 = DeformableConv(channel, channel, kernel_size=(3, 1), padding=(1, 0))

        self.bn_3x3 = nn.BatchNorm2d(num_features=channel, affine=True)
        self.bn_3x1 = nn.BatchNorm2d(num_features=channel, affine=True)
        self.bn_1x3 = nn.BatchNorm2d(num_features=channel, affine=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, data):

        fuseData = self.rDcn(data)
        fuseData = self.bn_3x3(fuseData)

        fuseData += self.bn_1x3(self.deform_conv_1x3(data))
        fuseData += self.bn_3x1(self.deform_conv_3x1(data))

        return self.relu(fuseData)


class ChannelFuse(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelFuse, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, data):
        
        b, c, _, _ = data.size()
        y = self.avg_pool(data).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return data * y.expand_as(data)


class FeatureFuse(nn.Module):
    def __init__(self, channels):
        super(FeatureFuse, self).__init__()


        self.spatialFuse = SpatialFuse(channels)
        self.channelFuse = ChannelFuse(channels)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, data):

        channelFuseData = self.channelFuse(data)
        spatialFuseData = self.spatialFuse(data)


        out = channelFuseData + spatialFuseData
        
        out = self.channel_shuffle(out, 2)

        return out


def get_norm(norm, Channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
        Channels: 输入的通道数

    Returns:
        nn.Module or None: the normalization layer
    """

    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None

        norm = {
            "BN": nn.BatchNorm2d,
            "SyncBN": nn.SyncBatchNorm,
            "GN": lambda channels: nn.GroupNorm(32, channels),
        }[norm]
    return norm(Channels)


class Conv(nn.Module):

    def __init__(self, inc, outc, kernel_size=1, padding=0, Norm="GN"):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, padding=padding)
        self.norm = get_norm(Norm, outc)
        self.activation = F.relu

    def forward(self, x):

        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class CSPModule(nn.Module):

    def __init__(self, in_channel, k, lastChange=False):
        super(CSPModule, self).__init__()
        self.k = k
        self.lastChange = lastChange

        self.ds_1 = nn.ModuleList([Conv(in_channel, in_channel // (k ** 2), kernel_size=4, padding=0, Norm="SyncBN")])

        self.conv1x1 = nn.Conv2d(in_channel // (k ** 2), in_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, data):
        _, c, h, w = data.shape
        nh, nw = 8, 8
        ph, pw = h // nh, w // nw
        upsample = nn.Upsample(size=(2 * h, 2 * w), mode='bilinear', align_corners=False)
        rh = rw = self.k
        pad_w = [rw // 2 - 1, rw // 2] if rw % 2 == 0 else [rw // 2] * 4
        pad_h = [rh // 2 - 1, rh // 2] if rh % 2 == 0 else [rh // 2] * 4
        context = F.pad(data, pad_w + pad_h)

        context = self.ds_1[0](context)

        context = upsample(context)

        context = self.conv1x1(context)

        return context

class JointData(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(JointData, self).__init__()

        self.linear = Conv(in_channel, out_channel, kernel_size=1, Norm="GN")
        self.linear_fuse = Conv(out_channel * 2, out_channel, kernel_size=1, Norm="GN")

    def forward(self, data1, data2):
        data1 = self.linear(data1)
        data1 = F.interpolate(data1, size=data2.size()[2:], mode='nearest')
        data = self.linear_fuse(torch.cat([data1, data2], dim=1))
        return data
