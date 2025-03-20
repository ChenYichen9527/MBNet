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



class HQFset(Dataset):
    def __init__(self, data_root, is_training):
        self.file_root = data_root
        self.training = is_training

        self.imgpath = []
        self.gtpath = []
        self.eventpath=[]
        self.event_root = os.path.join(self.file_root, 'events')
        self.image_root = os.path.join(self.file_root, 'blur')
        self.gt_root = os.path.join(self.file_root, 'images')

        for file in os.listdir(self.image_root):
            self.eventfile = os.path.join(self.event_root, file)
            self.imagefile = os.path.join(self.image_root, file)
            self.gt_file   = os.path.join(self.gt_root, file)

            for img in os.listdir(self.imagefile):
                index = int(img[:-4])
                self.imgpath.append( os.path.join(self.imagefile,img))
                self.gtpath.append(os.path.join(self.gt_file,img))
                self.eventpath.append([
                                        os.path.join(self.eventfile,str(index-3).zfill(8)+'.npy'),
                                       os.path.join(self.eventfile,str(index-2).zfill(8)+'.npy'),
                                       os.path.join(self.eventfile,str(index-1).zfill(8)+'.npy'),
                                       os.path.join(self.eventfile,str(index).zfill(8)+'.npy'),
                                       os.path.join(self.eventfile,str(index+1).zfill(8)+'.npy'),
                                       os.path.join(self.eventfile,str(index+2).zfill(8)+'.npy')]
                                       )
           

    def __getitem__(self, index):

        evnpath= self.eventpath[index]
        img2path = self.imgpath[index]
        img_gt_path = self.gtpath[index]

        Im2 = cv.imread(img2path)
        Igt = cv.imread(img_gt_path)

        # crop an image patch
        h, w, _ = Im2.shape
        img2 = Im2
        img_gt = Igt

        events=[]
        for i,p in enumerate(evnpath):
            

            events.append(np.load(p))
            if i==2:
                if events[-1].shape[0]!=0:
                    middletime = events[-1][:, 2:3].max()
                elif events[-2].shape[0]!=0:
                    middletime = events[-2][:, 2:3].max()
                else:
                    middletime = events[-3][:, 2:3].max()
        eventpoints = np.vstack(events)

        alltime = eventpoints[:, 2:3]
        if alltime.shape[0] > 1:
            # timestamp normalization
            tm =(middletime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5)
            alltime = (alltime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5) 
        # print(tm)
        x = (eventpoints[:, 0:1]) 
        y = (eventpoints[:, 1:2]) 
        p = eventpoints[:, 3:]

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
        # img1 = torch.Tensor(img1.copy()).float().permute(2, 0, 1) / 255.0
        img2 = torch.Tensor(img2.copy()).float().permute(2, 0, 1) / 255.0
        # img3 = torch.Tensor(img3.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(img_gt.copy()).float().permute(2, 0, 1) / 255.0

        return  img2, img2, img2,img_gt, left_reverse_vol,left_vol,right_vol


    def __len__(self):
        return len(self.imgpath)

if __name__=='__main__':
    Dataset= HQFset(r'E:\All_Dataset\HQF11\HQF',is_training=False)
    testloader = DataLoader(Dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    for i, img in enumerate(testloader):
        print()
