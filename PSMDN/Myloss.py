import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()  # 初始化，继承nn.module父类

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k


class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E


class L_exp(nn.Module):

    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class L_vs(nn.Module):
    def __init__(self, patch_size):
        super(L_vs, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, x, eps=1e-6):
        b, c, h, w = x.shape  # x [0, 255]
        Max, _ = torch.max(x, dim=1, keepdim=True)
        Min, _ = torch.min(x, dim=1, keepdim=True)

        one = torch.ones_like(Max).cuda()
        Max = torch.where(Max <= 0, one * 0.00392156862745, Max)

        value = Max
        saturation = (Max - Min) / Max

        result = value - saturation
        r = self.pool(result)

        # mean_exp = torch.mean(x, [1, 2, 3], keepdim=True) * 0.00392156862745
        # mean_exp = (1 - 0.5 * mean_exp) * mean_exp
        #
        # sigmoid = torch.exp(5 * (r - mean_exp)) / (1 + torch.exp(5 * (r - mean_exp)))
        #
        # re = sigmoid * torch.abs(r)

        e = 2.71828
        sigma = torch.exp(3 * r) / e**3

        re = torch.abs(r) * sigma
        loss = torch.mean(re, [1, 2, 3])

        return loss


def hue(img, epsilon=1e-10):
    img = img.float() * 255
    r, g, b = torch.split(img, 1, dim=1)

    max, arg_max = torch.max(img, dim=1, keepdim=True)
    min, arg_min = torch.min(img, dim=1, keepdim=True)

    max_min = max - min + epsilon

    h1 = 60.0 * (g - b) / max_min   # R
    h2 = 60.0 * (b - r) / max_min + 120.0  # G
    h3 = 60.0 * (r - g) / max_min + 240.0  # B

    h = torch.stack((h1, h2, h3), dim=1).gather(dim=1, index=arg_max.unsqueeze(0)).squeeze(0)
    return h


class L_hue(nn.Module):
    def __init__(self, batch_size):
        super(L_hue, self).__init__()
        self.pool = nn.AvgPool2d(batch_size)

    def forward(self, haze, dehaze):
        h1 = hue(haze)
        h2 = hue(dehaze)
        hue_haze = self.pool(h1)
        hue_dehaze = self.pool(h2)

        k = torch.mean(torch.pow(hue_haze - hue_dehaze, 2)) / 3600
        return k


class L_con(nn.Module):
    def __init__(self, batch_size):
        super(L_con, self).__init__()
        self.pool = nn.AvgPool2d(batch_size)

    def forward(self, dehaze, eps=1e-10):
        b, c, h, w = dehaze.shape
        dehaze = self.pool(dehaze)

        background = torch.mean(dehaze, dim=1, keepdim=True) + eps
        r_background = torch.abs(dehaze[:, 0, :, :] - background) / background
        g_background = torch.abs(dehaze[:, 1, :, :] - background) / background
        b_background = torch.abs(dehaze[:, 2, :, :] - background) / background

        loss = -1 * torch.mean(r_background + g_background + b_background, [1, 2, 3])

        return loss, background, r_background, g_background, b_background


def value(img):
    img = img.float()
    b, c, h, w = img.shape  # x [0, 255]
    Max, _ = torch.max(img, dim=1, keepdim=True)
    Min, _ = torch.min(img, dim=1, keepdim=True)

    one = torch.ones_like(Max).cuda()
    Max = torch.where(Max <= 0, one * 0.00392156862745, Max)

    v = Max
    return v


class L_value(nn.Module):
    def __init__(self, batch_size):
        super(L_value, self).__init__()
        self.pool = nn.AvgPool2d(batch_size)

    def forward(self, haze, dehaze):
        value_haze = value(haze)
        value_dehaze = value(dehaze)

        k = torch.mean(torch.pow(value_haze - value_dehaze, 2))
        return k
