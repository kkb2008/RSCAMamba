from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.cityscapesFish import *
from geoseg.models.DCSwin import dcswin_small
from geoseg.models.newPyramidMamba import swinMamba_base, swinMamba_tiny, swinMamba_small, resMamba_34
from tools.utils import Lookahead
from tools.utils import process_model_params
import cv2
import torch.nn as nn
from functools import partial


# training hparam
max_epoch = 40
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 2
lr = 5e-4
weight_decay = 1e-4 
backbone_lr = 7.5e-5
backbone_weight_decay = 7.5e-5
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "cityFish-tiny-640-512crop-e40-mask2former"
weights_path = "model_weights/woodscape/{}".format(weights_name)
test_weights_name = "cityFish-tiny-640-512crop-e40-mask2former"
log_name = 'woodscape/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1

pretrained_ckpt_path = None # the path for the pretrained model weight

gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

depths = [2, 2, 6, 2] # swin-tiny

net = swinMamba_tiny(num_classes=num_classes)


loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.10, ignore_index=ignore_index), 
                 DiceLoss(smooth=0.10, ignore_index=ignore_index),
                 FocalLoss(alpha=0.40, gamma=3, ignore_index=ignore_index,), 2.0, 1.5, 1.5) 

use_aux_loss = False

# define the dataloader
def get_training_transform():
    
    train_transform = [
        
        albu.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
        albu.RandomRotate90(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
  
    crop_aug = Compose([RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.1], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


class ResizeToDivisor:
    def __init__(self, divisor=32, padding_value=0):
        self.divisor = divisor
        self.padding_value = padding_value

    def __call__(self, img, **kwargs):
        h, w = img.shape[:2]

        new_h = (h + self.divisor - 1) // self.divisor * self.divisor
        new_w = (w + self.divisor - 1) // self.divisor * self.divisor

        top_padding = 0
        bottom_padding = new_h - h
        left_padding = 0
        right_padding = new_w - w

        padded_image = cv2.copyMakeBorder(
            img,
            top_padding, bottom_padding, left_padding, right_padding,
            cv2.BORDER_CONSTANT,
            value=self.padding_value
        )

        return padded_image


class ResizeMaskToDivisor:
    def __init__(self, divisor=32):
        self.divisor = divisor

    def __call__(self, mask, **kwargs):
        h, w = mask.shape[:2]

        new_h = (h + self.divisor - 1) // self.divisor * self.divisor
        new_w = (w + self.divisor - 1) // self.divisor * self.divisor

        top_padding = 0
        bottom_padding = new_h - h
        left_padding = 0
        right_padding = new_w - w

        padded_mask = cv2.copyMakeBorder(
            mask,
            top_padding, bottom_padding, left_padding, right_padding,
            cv2.BORDER_CONSTANT,
            value=0
        )

        return padded_mask


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)

def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_test_transform():
    test_transform = [
        albu.Normalize()
    ]
    return albu.Compose(test_transform)


def test_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_test_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

train_dataset = CityScapesDataset(data_root='/root/autodl-tmp/cityscapesFisheye', mode='train',
                                 mosaic_ratio=0, transform=train_aug)


val_dataset = CityScapesDataset(data_root='/root/autodl-tmp/cityscapesFisheye', mode='val',transform=val_aug)

test_dataset = CityScapesDataset(data_root='/root/autodl-tmp/cityscapesFisheye',
                                transform=val_aug)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)


layerwise_params = {

    "backbone": {"lr": backbone_lr, "weight_decay": backbone_weight_decay},

    "backbone.patch_embed.norm": {"lr": 5e-6, "weight_decay": 0.0},
    "backbone.norm": {"lr": 5e-6, "weight_decay": 0.0},
    "absolute_pos_embed": {"lr": 5e-6, "weight_decay": 0.0},
    "relative_position_bias_table": {"lr": 5e-6, "weight_decay": 0.0},

    "query_embed": {"lr": backbone_lr, "weight_decay": 0.0},
    "query_feat": {"lr": backbone_lr, "weight_decay": 0.0},
    "level_embed": {"lr": backbone_lr, "weight_decay": 0.0},

}
layerwise_params.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': {"lr": 5e-6, "weight_decay": 0.0}
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
layerwise_params.update({
    f'backbone.stages.{stage_id}.downsample.norm': {"lr": 5e-6, "weight_decay": 0.0}
    for stage_id in range(len(depths) - 1)
})


net_params = process_model_params(net, layerwise_params=layerwise_params, lr_scaling=0.1)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
