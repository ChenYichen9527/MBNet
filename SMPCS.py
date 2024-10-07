import torch
import torch.nn as nn
# from SM import PointConv_SM, gumbel_softmax, batch_gather
from helper_tool import Plot
import matplotlib.pyplot as plt
import numpy as np
import copy
import myutils
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
import math
import torch.nn.init as init

def subsampling(fea, knn_idx, scale_factor):
    b, c, n = fea.shape
    sub_fea = batch_gather(fea, knn_idx[:, :n//scale_factor, :]).max(2)[0]
    return sub_fea


def upsampling(fea, knn_idx):
    up_fea = batch_gather(fea, knn_idx).squeeze(2)
    return up_fea



class PointConv_SM(nn.Module):
    def __init__(self, in_channels, out_channels, n_neighbor):
        super(PointConv_SM, self).__init__()
        self.in_channels = in_channels  
        self.out_channels = out_channels
        self.n_neighbor = n_neighbor

        # mode 1
        if n_neighbor > 1:
            self.conv_1x1 = nn.Conv1d(in_channels+3, out_channels, 1, bias=False)

            self.conv_dw = nn.Parameter(torch.randn(1, out_channels, 3, 3, 3))
            fan = n_neighbor
            gain = init.calculate_gain('relu', math.sqrt(5))
            std = gain / math.sqrt(fan)
            bound = math.sqrt(3.0) * std
            self.conv_dw.data.uniform_(-bound, bound)
        else:
            self.conv_1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)




    def forward(self, input):
        if self.n_neighbor > 1:
            rel_xyz, sample_xyz, fea, knn_idx = input  # x: B * C * N | knn_idx: B * N * K
            b, n, k = knn_idx.shape

            kernel = F.grid_sample(self.conv_dw.expand(b, -1, -1, -1, -1), sample_xyz,
                                   mode='nearest', padding_mode='border', align_corners=False).squeeze()
            kernel = kernel.view(b, self.out_channels, -1, n)                               # B * C_out * K * N
            # conv
            neighbor_fea = batch_gather(fea, knn_idx)
            neighbor_fea = self.conv_1x1(torch.cat([neighbor_fea, rel_xyz], 1).reshape(b, -1, k*n)).view(b, -1, k, n)
            out = (kernel * neighbor_fea).sum(2)

            return out

        else:
            out = self.conv_1x1(input)

            return out


def batch_gather(x, knn_idx):
    b = x.shape[0]  #8
    idx = torch.arange(b).to(x.device).view(-1, 1, 1).expand(-1, knn_idx.size(1), knn_idx.size(2))  #[8, 20000, 8]
    #torch.arange(16)=0,1,2,3,4,5,6,7,8,9,10,11, 12, 13, 14, 15]),变成[16,1,1]的维度,变成([16, 20000, 8])

    out = x[idx, :, knn_idx].permute(0, 3, 2, 1)             #([8, 3, 8, 20000]               # B * C * K * N2
            ##([16, 3, 8, 20000])
    return out



class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_neighbors,  bn=True, relu=True):
        super(BasicConv, self).__init__()
        self.pointCnov = nn.Sequential(
            PointConv_SM(in_channels, out_channels, n_neighbors)
        ) 

        
        self.ln=nn.LayerNorm(out_channels)
        self.lrelu= nn.Sequential()
        if relu:
            self.lrelu.append(nn.LeakyReLU(0.1, True))


        
    def forward(self, x):
        out = self.pointCnov(x)
        out = self.ln(out.transpose(-1,-2)).transpose(-1,-2)
        out = self.lrelu(out )

        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_neighbors, n_layers=4, radius=1.0, use_mask=True):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_neighbors = n_neighbors
        self.n_layers = n_layers
        self.radius = radius
        self.relu = nn.LeakyReLU(0.1, True)
        self.use_mask = use_mask
        self.tau=1
        self.nums=1
        # body

        self.block =  nn.Sequential(
                BasicConv(in_channels, out_channels, n_neighbors, bn=True, relu=True),
                BasicConv(out_channels, out_channels, n_neighbors, bn=True, relu=True),
                BasicConv(out_channels, out_channels, n_neighbors, bn=True, relu=True)
        )

    
            # tail
        self.tail = nn.Sequential(
            nn.Conv1d(out_channels * n_layers, out_channels, 1,bias=False),
            nn.LayerNorm(out_channels),   
        )

        # shortcut
        shortcut = []
        shortcutln = []
        if in_channels != out_channels:
            shortcut.append(nn.Conv1d(in_channels, out_channels, 1, bias=False))
            shortcutln.append(nn.LayerNorm(out_channels))
        self.shortcut = nn.Sequential(*shortcut)
        self.shortcutln = nn.Sequential(*shortcutln)


    def forward(self, input):
        xyz, fea, knn_idx = input    #[8, 3, 20000])  #[8, 32, 20000])  ([8, 20000, 8]

        neighbor_xyz = batch_gather(xyz, knn_idx)     #16, 3, 8, 20000                                   # B, 3, K, N
        rel_xyz = neighbor_xyz - xyz.unsqueeze(2)                                       # B, 3, K, N
        sample_xyz = rel_xyz / self.radius                                              # B, 3, K, N
        sample_xyz = sample_xyz.permute(0, 2, 3, 1).unsqueeze(-2)

        # body
        buffer=[]
        init_fea = fea
        for i in range(  self.n_layers) :
            fea = self.block[i]( [rel_xyz, sample_xyz, fea, knn_idx])
            buffer.append(fea)
        # tail
        out_1 = self.tail[1](self.tail[0](torch.cat(buffer, 1)).transpose(-1,-2)).transpose(-1,-2)
        out_2 = self.shortcutln(self.shortcut(init_fea).transpose(-1,-2)).transpose(-1,-2)
        
        out = self.relu (out_1+out_2)
        return out



class SMPCnet(nn.Module):
    def __init__(self, d_in, n_neighbors=16, n_blocks=8):   #d_in=4
        super(SMPCnet, self).__init__()
        self.n_neighbors = n_neighbors
        self.n_blocks = n_blocks
        self.relu = nn.LeakyReLU(0.1, True)
        # initial
        self.init = BasicConv(d_in, 32, 1, bn=True, relu=True)

        # body
        body = [
            BasicBlock(d_in, 32, n_neighbors, n_layers=2, radius=1/4),
            BasicBlock(32, 64, n_neighbors, n_layers=2, radius=2/4),
            BasicBlock(64, 128, n_neighbors, n_layers=2, radius=3/4),
            BasicConv(128+64, 64, 1, bn=True, relu=True),
            BasicConv(64+32, 33, 1, bn=True, relu=False)
        ]
        self.body = nn.Sequential(*body)

    def forward(self, input):
        x, knn_idx = input          #([8, 20000, 4])  #9组邻点的序号
        x = x.transpose(-1, -2)     #([8, 4, 20000])
        xyz = x[:, :3, :]
        b, _, n = xyz.shape

        fea1 = self.body[0]([xyz, x[:,-1:,:],      knn_idx[0]])   
        
        xyz_1 = xyz[:, :,::4]                     
        fea2 = self.body[1]([xyz_1, subsampling(fea1, knn_idx[0], 4), knn_idx[1]])          # 1/4


        xyz_2 = xyz_1[:, :,::4]  
        fea3 = self.body[2]([xyz_2, subsampling(fea2, knn_idx[1], 4), knn_idx[2]])         # 1/16
                                              # 1/16

        fea8 = torch.cat([upsampling(fea3, knn_idx[3]), fea2], 1)
        fea8 = self.body[3](fea8)                                                        # 1/4

        fea9 = torch.cat([upsampling(fea8, knn_idx[4]), fea1], 1)    #4, 96, 4608
        fea9 = self.body[4](fea9) 
        
        out = self.relu(fea9[:,:32,...])# 1
        ratio = torch.sigmoid(fea9[:,-1:,...])
        return out, ratio
