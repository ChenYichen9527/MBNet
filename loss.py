# Sourced from https://github.com/myungsub/CAIN/blob/master/loss.py, who sourced from https://github.com/thstkdgus35/EDSR-PyTorch/tree/master/src/loss
# Added Huber loss in addition.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_msssim
from model import optical_flow_warp


import torch.optim as optim



def flow_smooth_loss(flow):
    loss = (flow[:, :, :-1, :-1] - flow[:, :, 1:, :-1]).abs() + (flow[:, :, :-1, :-1] - flow[:, :, :-1, 1:]).abs()

    return loss.mean()

# Wrapper of loss functions



# class Loss(nn.modules.loss._Loss):
#     def __init__(self):
#         super(Loss, self).__init__()

#         self.loss = []
#         self.loss_module = nn.ModuleList()

#         self.loss = nn.L1Loss()

#     def forward(self, sr, hr, img1, img2, flow1, flow2):
#         loss_sr = self.loss(sr, hr)
#         loss_flow1 = self.loss(optical_flow_warp(img1, flow1), hr) + 0.1 * flow_smooth_loss(flow1)
#         loss_flow2 = self.loss(optical_flow_warp(img2, flow2), hr) + 0.1 * flow_smooth_loss(flow2)

#         return loss_sr, 0.5 * (loss_flow1 + loss_flow2)


class Loss(nn.modules.loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()

        self.loss = nn.L1Loss()

    def forward(self, sr, hr):
        loss_sr = self.loss(sr, hr)

        return loss_sr