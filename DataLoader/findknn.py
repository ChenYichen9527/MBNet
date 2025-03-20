
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from helper_tool import Plot
import copy

from myutils import *


def findknn(events):
    """
    点云的KNN
    """
    xyt_1 = torch.cat([events[:,0:1]/2, events[:,1:3] ], 1)  # events[:,0:3]


    neighbor_idx_1 = knn(xyt_1, xyt_1, 16,omp=True).astype(np.int32)
    neighbor_idx_1 = neighbor_idx_1.reshape(-1)
    neighbor_idx_1 = neighbor_idx_1[:neighbor_idx_1.shape[0] // 2].reshape(-1, 8)
    neighbor_idx_1 = torch.from_numpy(neighbor_idx_1).long()


    xyt_2 = xyt_1[::4,:]

    neighbor_idx_2 = knn(xyt_2, xyt_2, 16,omp=True).astype(np.int32)
    neighbor_idx_2 = neighbor_idx_2.reshape(-1)
    neighbor_idx_2 = neighbor_idx_2[:neighbor_idx_2.shape[0] // 2].reshape(-1, 8)
    neighbor_idx_2 = torch.from_numpy(neighbor_idx_2).long()

   
    xyt_3 = xyt_2[::4, :]
    
    
    # xyt_3 = xyt_2[::4, :]
    neighbor_idx_3 = knn(xyt_3, xyt_3, 16,omp=True).astype(np.int32)
    neighbor_idx_3 = neighbor_idx_3.reshape(-1)
    neighbor_idx_3 = neighbor_idx_3[:neighbor_idx_3.shape[0] // 2].reshape(-1, 8)
    neighbor_idx_3 = torch.from_numpy(neighbor_idx_3).long()

    # display_neighbor(neighbor_idx_3,xyt_3)

    
    ## 1/16 -> 1/4
    up_idx_4 = knn(xyt_3, xyt_2, 2, omp=True).astype(np.int32)
    up_idx_4 = up_idx_4.reshape(-1)
    up_idx_4 = up_idx_4[:up_idx_4.shape[0] // 2].reshape(-1, 1)
    up_idx_4 = torch.from_numpy(up_idx_4).long()
    
    ## 1/4 -> 1
    up_idx_5 = knn(xyt_2, xyt_1, 2, omp=True).astype(np.int32)
    up_idx_5 = up_idx_5.reshape(-1)
    up_idx_5 = up_idx_5[:up_idx_5.shape[0] // 2].reshape(-1, 1)
    up_idx_5 = torch.from_numpy(up_idx_5).long()



    return [neighbor_idx_1, neighbor_idx_2, neighbor_idx_3, up_idx_4, up_idx_5]