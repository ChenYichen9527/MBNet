import os
import time
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import math
import matplotlib.pyplot as plt
from DataLoader.findknn import findknn

from helper_tool import Plot
from myutils import *
class Goprotestdataset_sf(Dataset):
    def __init__(self, data_root, is_training):
        self.data_root = data_root
        self.training = is_training
        if is_training:
            self.file_root = os.path.join(self.data_root, 'train')
        else:
            self.file_root = os.path.join(self.data_root, 'val')

        self.imageslist = []
        self.eventslist = []
        self.event_root = os.path.join(self.file_root, 'event')
        self.image_root = os.path.join(self.file_root, 'image_new')
        self.image_file = os.path.join(self.file_root, 'ave_bicubic')
        self.gt_file = os.path.join(self.file_root, 'sharp_gray_lr')
        self.filelength = len(os.listdir(self.event_root))

        for npyfile in os.listdir(self.event_root):
            eventfiles = os.path.join(self.event_root, npyfile)
            self.eventslist.append(eventfiles)  # here

        for txtf in os.listdir(self.image_root):
            imgslines = []
            imgsfiles = os.path.join(self.image_root, txtf)
            with open(imgsfiles, 'r') as f_img:
                for line in f_img.readlines():
                    line = line.strip("\n").split()
                    imgslines.append(line)

            self.imageslist.append(imgslines)  # here
            self.eventsdata = []

    def __getitem__(self, index):
        # t1=time.time()

        file_i = index // 94
        lines_i = index % 94  +3

        time1 = float(self.imageslist[file_i][lines_i][0])
        middletime =float( self.imageslist[file_i][lines_i+1][0])
        time2 = float(self.imageslist[file_i][lines_i+2][0])

        image1 = self.imageslist[file_i][lines_i-1][1]
        image2 = self.imageslist[file_i][lines_i][1]
        image3 = self.imageslist[file_i][lines_i+1][1]

        imagegt = self.imageslist[file_i][lines_i+1][1]
        self.eventsdata = np.load(self.eventslist[file_i])

        img1path = os.path.join(self.image_file, image1)
        img2path = os.path.join(self.image_file, image2)
        img3path = os.path.join(self.image_file, image3)
        img_gt_path = os.path.join(self.gt_file, imagegt)

        Im1 = cv.imread(img1path)
        Im2 = cv.imread(img2path)
        Im3 = cv.imread(img3path)
        Igt = cv.imread(img_gt_path)

        # crop an image patch
        h, w, _ = Im1.shape
        # size = [64, 64]
        # h_start = math.floor((h - size[0]) * random.random())
        # h_end = h_start + size[0]
        # w_start = math.floor((w - size[1]) * random.random())
        # w_end = w_start + size[1]
        # img1 = Im1[h_start:h_end, w_start:w_end, :]
        # img2 = Im2[h_start:h_end, w_start:w_end, :]
        # img3 = Im3[h_start:h_end, w_start:w_end, :]
        # img_gt = Igt[h_start:h_end, w_start:w_end, :]
        img1 = Im1
        img2 = Im2
        img3 = Im3
        img_gt = Igt

        # crop an event patch
        tp = (self.eventsdata[:, 0] > float(time1)) & (self.eventsdata[:, 0] < float(time2))
        eventpoints = self.eventsdata[tp, :]
        # wp = (eventpoints1[:, 1] >= float(w_start)) & (eventpoints1[:, 1] < float(w_end))
        # hp = (eventpoints1[:, 2] >= float(h_start)) & (eventpoints1[:, 2] < float(h_end))
        # eventpoints = eventpoints1[wp & hp, :]
        # np.random.shuffle(eventpoints)
        # print(eventpoints.shape[0])

        alltime = eventpoints[:, 0:1]
        if alltime.shape[0] > 1:
            # timestamp normalization
            tm = (middletime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5) 
            alltime = (alltime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5) 

        x = (eventpoints[:, 1:2]) 
        y = (eventpoints[:, 2:3]) 
        p = eventpoints[:, 3:] *2 -1

        eventpoints = np.concatenate([ alltime, x, y, p], 1)
        eventpoints = torch.Tensor( eventpoints)


        left_events,right_events,mask=e_split(eventpoints,tm)
        left_reverse_events=e_reverse(left_events)
        # left_events=e_reverse(left_events)

        # t2 = time.time()
        index=[7,6,5,4,3,2,1,0]
        left_reverse_vol = to_voxel_grid(left_reverse_events, h=h, w=w)
        pos_left = left_events[left_events[:,-1]==1]
        neg_left = left_events[left_events[:,-1]==-1]
        pos_left_vol = to_voxel_grid(np.array(pos_left), h=h, w=w)
        neg_left_vol = to_voxel_grid(np.array(neg_left), h=h, w=w)
        pos_left_vol = pos_left_vol[index,:,:]
        neg_left_vol = neg_left_vol[index,:,:]
        left_vol = torch.cat([pos_left_vol,neg_left_vol],0)

        pos_right = right_events[right_events[:,-1]==1]
        neg_right = right_events[right_events[:,-1]==-1]
        # right_vol = to_voxel_grid(np.array(right_events), h=size[0], w=size[1])
        pos_right_vol = to_voxel_grid(np.array(pos_right), h=h, w=w)
        neg_right_vol = to_voxel_grid(np.array(neg_right), h=h, w=w)
        right_vol = torch.cat([pos_right_vol,neg_right_vol],0)


        # augmentation
        # if self.training:
        #     img1, img2, img_gt, events = self._augmentation(img1, img2, img_gt, events)

        # to Tensor
        img1 = torch.Tensor(img1.copy()).float().permute(2, 0, 1) / 255.0
        img2 = torch.Tensor(img2.copy()).float().permute(2, 0, 1) / 255.0
        img3 = torch.Tensor(img3.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(img_gt.copy()).float().permute(2, 0, 1) / 255.0

        return img1, img2, img3,img_gt, left_reverse_vol,left_vol,right_vol


    def __len__(self):
        return self.filelength * 94

