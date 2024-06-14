import torch
import torch.nn as nn


class L_con(nn.Module):
    def __init__(self):
        super(L_con, self).__init__()

    def forward(self, dehaze):
        b, c, h, w = dehaze.shape

        background = torch.mean(dehaze, dim=1, keepdim=True)
        r_background = torch.abs(dehaze[:, 0, :, :] - background) / background
        g_background = torch.abs(dehaze[:, 1, :, :] - background) / background
        b_background = torch.abs(dehaze[:, 2, :, :] - background) / background

        loss = -1 * torch.mean(r_background + g_background + b_background)

        return loss
