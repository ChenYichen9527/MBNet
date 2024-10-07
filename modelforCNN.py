from asyncio import events
import math
from tkinter import E
from turtle import forward
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import size_adapter
from torchvision import transforms




class ResB(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2),
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.body(x) + x)

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1,padding=0),
            nn.ReLU(True))
        self.AFFblock = nn.Sequential(
            ResB(out_channel),
            ResB(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1,padding=1)
        )
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU(True)
        
    def forward(self, x1, x2, x4):
        x =self.conv1(torch.cat([x1, x2, x4], dim=1))
        return self.conv2(self.relu(self.AFFblock(x)+x))

class Unet(nn.Module):
    def __init__(self,input_dim=3):
        super().__init__()
        self._size_adapter = size_adapter.SizeAdapter(minimum_size=16)
        self.input_dim = input_dim
        self.relu = nn.ReLU(True)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            ResB(32)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            ResB(48)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            ResB(64)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 72, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            ResB(72)
        )

        self.dconv1 =   nn.Sequential(
            ResB(72),
            nn.Conv2d(72, 72, 3, 1, 1)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(72+64, 64, 3, 1, 1),
            nn.ReLU(True),
            ResB(64)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64+48, 48, 3, 1, 1),
            nn.ReLU(True),
            ResB(48)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(48+32, 32, 3, 1, 1),
            nn.ReLU(True),
            ResB(32)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        
        self.body5 = nn.Sequential(
            ResB(64),
            nn.Conv2d(64, 64, 3, 1, 1))
        self.conv5=nn.Conv2d(64,  64,  3, 1, 1)
        
        self.body6 = nn.Sequential(
            ResB(48),
            nn.Conv2d(48, 48, 3, 1, 1))
        self.conv6=nn.Conv2d(48,  48,  3, 1, 1)

        self.body = nn.Sequential(
            ResB(32),
            nn.Conv2d(32, 32, 3, 1, 1)
        )
        self.tail=nn.Conv2d(32,  32,  3, 1, 1)


    def forward(self, image1,image2,image3,fea15,fea16,fea17,fea25,fea26,fea27,args):
        image1 = self._size_adapter.pad(image1)
        image2 = self._size_adapter.pad(image2)
        image3 = self._size_adapter.pad(image3)

        if args.sequences==1:
            fea1_1 = self.encoder1(image2) 
        else:
            fea1_1 = self.encoder1(torch.cat([image1,image2,image3],1))   #16
        # fea1_1 = self.encoder1(image2)
        # fea3_1 = self.encoder1(image3)


        fea1_2 = self.encoder2(fea1_1)   #32
        # fea2_2 = self.encoder2(fea2_1)
        # fea3_2 = self.encoder2(fea3_1)

        fea1_3 = self.encoder3(fea1_2)  #48
        # fea2_3 = self.encoder3(fea2_2)
        # fea3_3 = self.encoder3(fea3_2)
 
        fea1_4 = self.encoder4(fea1_3)  #64
        # fea2_4 = self.encoder4(fea2_3)
        # fea3_4 = self.encoder4(fea3_3)

        # fea4 = self.dconv1(torch.cat([fea1_4,fea2_4,fea3_4],1))  #128
        fea4 = self.dconv1(fea1_4)  #128

        # fea5 = self.decoder1(torch.cat([self.upsample(fea4), fea1_3, fea2_3,fea3_3], 1))
        fea5 = self.decoder1(torch.cat([self.upsample(fea4), fea1_3], 1))
        fea5 = fea5 - fea15 + fea25
        fea5 = self.conv5(self.relu(self.body5(fea5)  + fea5 ))

        fea6 = self.decoder2(torch.cat([self.upsample(fea5), fea1_2], 1))
        fea6 = fea6 - fea16 + fea26
        fea6 = self.conv6(self.relu(self.body6(fea6)  + fea6 ))


        fea7 = self.decoder3(torch.cat([self.upsample(fea6), fea1_1], 1))
        fea7 = fea7 - fea17 + fea27
        fea_7=self._size_adapter.unpad(fea7)
        im_out= self.tail(self.relu(self.body(fea7)  + fea7 ))




        return self._size_adapter.unpad(im_out)
   
    







# class Unet1(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self._size_adapter = size_adapter.SizeAdapter(minimum_size=16)
#         self.relu = nn.ReLU(True)
#         self.encoder1 = nn.Sequential(
#             nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=2,dilation=2),
#             nn.ReLU(True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2,dilation=2),
#             nn.ReLU(True)
#             )
#         self.conv1=  nn.Sequential(
#             ResB(32), 
#             nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(True),
#         )
#         self.conv11 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

#         self.encoder2 = nn.Sequential(
#             nn.Conv2d(10, 48, kernel_size=3, stride=2, padding=2,dilation=2),
#             nn.ReLU(True),
#             nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=2,dilation=2),
#             nn.ReLU(True)
#         )

#         self.conv2=  nn.Sequential(
#             ResB(48), 
#             nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(True),
#         )
#         self.conv22 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)


#         self.encoder3 = nn.Sequential(
#             nn.Conv2d(10, 64, kernel_size=3, stride=2, padding=2,dilation=2),
#             nn.ReLU(True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=2,dilation=2),
#             nn.ReLU(True)
#         )
#         self.conv3=  nn.Sequential(
#             ResB(64), 
#             nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(True),
#         )
#         self.conv33 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)




        
        
#     def forward(self,left_vol):
#         vol1 = self._size_adapter.pad(left_vol)

#         fea1 = self.encoder1(vol1)
#         fea1 = self.conv11(self.conv1(fea1)+fea1)
        
#         fea2 = self.encoder2(vol1)   #32
#         fea2 = self.conv22(self.conv2(fea2)+fea2)

#         fea3 = self.encoder3(vol1)  #48
#         fea3 = self.conv33(self.conv3(fea3)+fea3)
#         # fea4 = self.encoder4(fea3)  #64
#         # fea4 = self.dconv1(fea4)  #128

#         return fea3,fea2,fea1



class Unet1(nn.Module):
    def __init__(self):
        super().__init__()
        self._size_adapter = size_adapter.SizeAdapter(minimum_size=16)
        self.relu = nn.ReLU(True)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            ResB(32),
            
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            ResB(48)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            ResB(64)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 72, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            ResB(72)
        )

        # self.dconv1 =   nn.Sequential(
        #     ResB(72)
        # )

        self.dconv1 =   nn.Sequential(
            ResB(72),
            ResB(72),
            nn.Conv2d(72, 72, 3, 1, 1)
        )
        self.relu = nn.ReLU(True)
        self.dconv2 =nn.Conv2d(72, 72, 3, 1, 1)
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(72+64, 64, 3, 1, 1),
            nn.ReLU(True),
            ResB(64)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64+48, 48, 3, 1, 1),
            nn.ReLU(True),
            ResB(48)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(48+32, 32, 3, 1, 1),
            nn.ReLU(True),
            ResB(32)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.body = nn.Sequential(
        #     ResB(48),
        #     ResB(48),
        #     nn.Conv2d(48, 48, 3, 1, 1)
        # )
        # self.tail=nn.Conv2d(48,  3,  3, 1, 1)
        self.aff1=AFF(144,64)
        self.aff2=AFF(144,48)
        self.aff3=AFF(144,32)
    def forward(self,left_vol):
        vol1 = self._size_adapter.pad(left_vol)

        fea1 = self.encoder1(vol1)  
        fea2 = self.encoder2(fea1)   #32
        fea3 = self.encoder3(fea2)  #48
        fea4 = self.encoder4(fea3)  #64
        # fea4 = self.dconv1(fea4)  #128
        fea4 = self.dconv2(self.relu(self.dconv1(fea4)+fea4) ) #128

        fea3_aff = self.aff1(fea3,F.interpolate(fea2, scale_factor=0.5, mode='bilinear'),F.interpolate(fea1, scale_factor=0.25, mode='bilinear'))
        fea5 = self.decoder1(torch.cat([self.upsample(fea4), fea3_aff], 1))

        fea2_aff =  self.aff2(self.upsample(fea3),fea2,F.interpolate(fea1, scale_factor=0.5, mode='bilinear'))
        fea6 = self.decoder2(torch.cat([self.upsample(fea5), fea2_aff], 1))

        fea1_aff =  self.aff3(F.interpolate(fea3, scale_factor=4, mode='bilinear'),self.upsample(fea2),fea1)
        fea7 = self.decoder3(torch.cat([self.upsample(fea6), fea1_aff], 1))


    
        return fea5,fea6,fea7
