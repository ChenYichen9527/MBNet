from mimetypes import init
from turtle import left
import matplotlib
from torch.nn.functional import bilinear
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from cv2 import resizeWindow, transpose
from numpy import reshape
from SMPCS import SMPCnet
from modelforCNN import *
from myutils import *
import pylab
# import Unet
class Weight_Conv (nn.Module):
    def __init__(self,input_dim = 32):
        
        super().__init__()
        self.dim = 64
        self.init_conv = nn.Conv2d(input_dim, self.dim, 3, 1, 1)
        self.resb = ResB(self.dim)
        self.tailconv = nn.Conv2d(self.dim, 1, 3, 1, 1)
    def forward(self,vol):
        output = self.tailconv(self.resb(self.init_conv(vol)))
        return F.sigmoid(output)
class Pos2Weight(nn.Module):
    def __init__(self,inC, kernel_size=7, outC=3):
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(4,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,self.kernel_size*self.kernel_size*self.inC*self.outC)
        )
    def forward(self,x):

        output = self.meta_block(x)
        return output

class ResB3d(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResB3d, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size, 1, kernel_size//2),
            nn.ReLU(True),
            nn.Conv3d(channels, channels, kernel_size, 1, kernel_size//2),
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.body(x) + x)



# def optical_flow_warp(image, image_optical_flow):
#     """
#     Arguments
#         image_ref: reference images tensor, (b, c, h, w)
#         image_optical_flow: optical flow to image_ref (b, 2, h, w)
#     """
#     b, _, h, w = image.size()
#     grid = np.meshgrid(range(w), range(h))
#     grid = np.stack(grid, axis=-1).astype(np.float64)
#     grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
#     grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
#     grid = torch.Tensor(grid).to(image.device).unsqueeze(0)

#     flow_0 = torch.unsqueeze(image_optical_flow[:, 0, :, :] * 30 / (w - 1), dim=1)
#     flow_1 = torch.unsqueeze(image_optical_flow[:, 1, :, :] * 30 / (h - 1), dim=1)
    
#     grid = grid + torch.cat([flow_0, flow_1], 1).permute(0, 2, 3, 1)
#     output = F.grid_sample(image, grid, padding_mode='border')
#     return output

def optical_flow_warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output


"""
-----------------------主网络部分-------------------------------------------------------
"""
class Weight_Conv (nn.Module):
    def __init__(self,input_dim = 32):
        
        super().__init__()
        self.dim = 64
        self.init_conv = nn.Conv2d(input_dim, self.dim, 3, 1, 1)
        self.resb = ResB(self.dim)
        self.tailconv = nn.Conv2d(self.dim, 1, 3, 1, 1)
    def forward(self,vol):
        output = self.tailconv(self.resb(self.init_conv(vol)))
        return F.sigmoid(output)
    
class myNet(nn.Module):
    def __init__(self,input_dim=3):
        super().__init__()
        self.input_dim = input_dim
        self.img_Unet = Unet(self.input_dim)
        # self.vol_net_l=Unet1()
        self.vol_net_r=Unet1()
        # self.P2W = Pos2Weight(inC=32)
        # self.weight_conv = Weight_Conv()
        self.weight_conv = nn.Sequential(
                ResB(32),
                nn.Conv2d(32,32*3,3, 1, 1))
        # self.resconv=ResB(48)
    def forward(self, img1, img2, img3,left_reverse_vol,left_vol,right_vol,s,args):
        b, c, h, w = img1.shape
        meanval = img2.mean(-1).mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        img2 = img2-meanval
        # img1_fea = self.fea(img1)
        # img2_fea = self.fea(img2)
  
       
       
        
        
        e_mat =torch.cat([left_vol,right_vol],dim=1)
        dynamic_weight = self.weight_conv(e_mat)  #b,32*48,h,w
        dynamic_weight = dynamic_weight.view(b,-1,s*s*3,h,w)
        dynamic_weight =nn.Softmax(dim=1)(dynamic_weight)
        fea15,fea16,fea17 = self.vol_net_r(left_vol) 
        fea25,fea26,fea27 = self.vol_net_r(right_vol) 
        x = self.img_Unet(img1,img2,img3,fea15,fea16,fea17,fea25,fea26,fea27,args)    #可以尝试不同的光流网络
        # dynamic_weight = self.weight_conv( F.upsample(torch.cat([left_vol,right_vol],dim=1),[int(h*s),int(w*s)],mode='bilinear'))
  
        out = x.unsqueeze(2) * dynamic_weight
        
        output= (out.sum(1))+ meanval
        return output







