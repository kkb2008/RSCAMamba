import torch
from torch import nn
from geoseg.models.combin_module import Decoder
from geoseg.models.DCSwin import SwinTransformer
import torch.nn.functional as F
import timm

class PyramidMamba(nn.Module):
    def __init__(self,
                 encoder_channels=(128, 256, 512, 1024),
                 num_classes=10,
                 embed_dim=128,
                 depths=(2, 2, 18, 2),
                 num_heads=(4, 8, 16, 32),
                 decoder_channels=128,
                 last_feat_size=(30, 40),
                 frozen_stages=2):
        super(PyramidMamba, self).__init__()

        self.backbone = SwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                        frozen_stages=frozen_stages)

        self.decoder = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels,
                               num_classes=num_classes, last_feat_size=last_feat_size)

    def forward(self, x):
        
        data = self.backbone(x)

        x = self.decoder(data)
        x = F.interpolate(x, (966, 1280), mode='bilinear', align_corners=False)
        
        return x

class EfficientPyramidMamba(nn.Module):
    def __init__(self,
                 encoder_channels=(256, 512, 1024, 2048),
                 backbone_name='resnet50',
                 pretrained=True,
                 embed_dim=64,
                 num_classes=10,
                 decoder_channels=128,
                 last_feat_size=(30, 40)
                 ):
        super(EfficientPyramidMamba, self).__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32, out_indices=(1, 2, 3, 4), pretrained=pretrained)
        
        encoder_channels = self.backbone.feature_info.channels()
        
        self.decoder = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels,
                               num_classes=num_classes, last_feat_size=last_feat_size)

    def forward(self, x):
        data = self.backbone(x)
        x = self.decoder(data)

        return x


def swinMamba_base(pretrained=True, num_classes=4, weight_path='GeoSeg/geoseg/pretrain_weights/stseg_base.pth'):
    # pretrained weights are load from official repo of Swin Transformer
    model = PyramidMamba(encoder_channels=(128, 256, 512, 1024),
                         num_classes=num_classes,
                         embed_dim=128,
                         depths=(2, 2, 18, 2),
                         num_heads=(4, 8, 16, 32),
                         frozen_stages=2)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def swinMamba_small(pretrained=True, num_classes=4, weight_path='GeoSeg/geoseg/pretrain_weights/stseg_small.pth'):
    # pretrained weights are load from official repo of Swin Transformer
    model = PyramidMamba(encoder_channels=(96, 192, 384, 768),
                         num_classes=num_classes,
                         embed_dim=96,
                         depths=(2, 2, 18, 2),
                         num_heads=(3, 6, 12, 24),
                         frozen_stages=2)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def swinMamba_tiny(pretrained=True, num_classes=4, weight_path='GeoSeg/geoseg/pretrain_weights/stseg_tiny.pth'):
    # pretrained weights are load from official repo of Swin Transformer
    model = PyramidMamba(encoder_channels=(96, 192, 384, 768),
                         num_classes=num_classes,
                         embed_dim=96,
                         depths=(2, 2, 6, 2),
                         num_heads=(3, 6, 12, 24),
                         frozen_stages=2)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def resMamba_34(pretrained=True, num_classes=4, backbone_name='resnet34'):
    model = EfficientPyramidMamba(encoder_channels=(64, 128, 256, 512),
                                  backbone_name=backbone_name,
                                  pretrained=pretrained,
                                  embed_dim=64,
                                  num_classes=num_classes,
                                  decoder_channels=128)
    return model


def resMamba_101(pretrained=True, num_classes=4, backbone_name='resnet101'):
    model = EfficientPyramidMamba(encoder_channels=(256, 512, 1024, 2048),
                                  backbone_name=backbone_name,
                                  pretrained=pretrained,
                                  embed_dim=64,
                                  num_classes=num_classes,
                                  decoder_channels=128)
    return model
