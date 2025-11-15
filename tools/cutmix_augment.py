import torch
import numpy as np


class CutMix:
    def __init__(self, alpha=1.0, min_lam=0.3, max_lam=0.4):
        """
        初始化 CutMix 类

        :param alpha: Beta 分布的超参数，控制裁剪框的大小分布
        :param min_lam: lam 的最小值，用于裁剪框的大小控制
        :param max_lam: lam 的最大值，用于裁剪框的大小控制
        """
        self.alpha = alpha
        self.min_lam = min_lam
        self.max_lam = max_lam

    def rand_bbox(self, size, lam):
        """
        生成裁剪框的坐标

        :param size: 输入图像的大小，形状为 (batch_size, channels, height, width)
        :param lam: 从 Beta 分布采样得到的值，控制裁剪框的大小
        :return: 裁剪框的四个坐标 (bbx1, bby1, bbx2, bby2)
        """
        if len(size) == 4:
            W = size[2]
            H = size[3]
        elif len(size) == 3:
            W = size[0]
            H = size[1]
        else:
            raise Exception

        cut_rat = np.sqrt(1. - lam)  # 裁剪比例
        cut_w = np.int(W * cut_rat)  # 裁剪框的宽度
        cut_h = np.int(H * cut_rat)  # 裁剪框的高度

        # 随机选择裁剪框的中心点
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # 计算裁剪框的左上角和右下角坐标，确保裁剪框不会超出边界
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def cutmix(self, data, target):
        """
        进行 CutMix 数据增强

        :param data: 输入图像数据，形状为 (batch_size, channels, height, width)
        :param target: 输入标签数据，形状为 (batch_size, channels, height, width)
        :return: 增强后的图像和标签
        """
        # 随机生成一个索引
        indices = torch.randperm(data.size(0))

        # 从 Beta 分布采样 lam，控制裁剪框大小
        lam = np.clip(np.random.beta(self.alpha, self.alpha), self.min_lam, self.max_lam)

        # 生成裁剪框
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.size(), lam)

        # 克隆原始图像和标签
        new_data = data.clone()
        new_target = target.clone()

        # 用随机选择的图像和标签填充裁剪框区域
        new_data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
        new_target[:, :, bbx1:bbx2, bby1:bby2] = target[indices, :, bbx1:bbx2, bby1:bby2]

        return new_data, new_target
