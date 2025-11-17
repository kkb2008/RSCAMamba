import torch
from torch import nn
import torch.nn.functional as F
from geoseg.models import feature_extraction


class KModule(nn.Module):
    def __init__(self, channels, nextChannels, act=True):
        super(KModule, self).__init__()
        self.likeSe = feature_extraction.FeatureFuse(channels)
        self.vw = feature_extraction.CSPModule(channels, 4, lastChange=True)
        self.conv = nn.Conv2d(channels, nextChannels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(nextChannels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, data1, data2):
        outData = self.likeSe(data1)
        outData = self.vw(outData)
        outData = self.conv(outData)
        outData = self.act(self.bn(outData))
        target_height, target_width = data2.shape[2:]
        outData = F.interpolate(outData, size=(target_height, target_width), mode='bilinear', align_corners=False)
        outData = outData + data2
        return outData
        

class KIntegration(nn.Module):
    def __init__(self, channels5, channels4, channels3):
        super(KIntegration, self).__init__()
        self.fuse5and4 = KModule(channels5, channels4)
        self.fuse4and3 = KModule(channels4, channels3)
        self.likeSe = feature_extraction.FeatureFuse(channels3)

    def forward(self, data):
        res2, res3, res4, res5 = data
        _, _, H, W = res2.shape
        out = []
        outData4 = self.fuse5and4(res5, res4)
        outData3 = self.fuse4and3(outData4, res3)
        outData2 = self.likeSe(outData3)
        for d in [outData2, outData3, outData4]:
            pool_out = F.interpolate(d, (H, W), mode='bilinear', align_corners=False)
        out.append(pool_out)

        out = torch.cat(out, dim=1)
        return out
