from PIL import Image
import os
import torchvision.transforms.functional as F
import cv2 as cv
import torch
from pytorch_msssim import ssim_matlab as calc_ssim
from myutils import *
import myutils
savepath= r"/media/root/LENOVO_USB_HDD/Result/Deblur_Results/result/code_dn_v2/REBlur"
losses, psnrs, ssims = myutils.init_meters()
img_name='img_name'
gt_name='gt_name'
file =sorted(os.listdir(savepath))
for name in file:
    if name.endswith("0.png"):
        img_name =name
    if name.endswith("1.png"):
        gt_name =name
    if gt_name[:-5]==img_name[:-5]:
        img=  Image.open(os.path.join(savepath,img_name)).convert('L') 
        imggt=  Image.open(os.path.join(savepath,gt_name)).convert('L') 
        img = F.to_tensor( img)   
        imggt = F.to_tensor( imggt)   
        # img2 = torch.Tensor(img.copy()).float().permute(2, 0, 1) / 255.0
        # img_gt = torch.Tensor(imggt.copy()).float().permute(2, 0, 1) / 255.0
        myutils.eval_metrics( img.unsqueeze(0) , imggt.unsqueeze(0), psnrs, ssims)
  
print(" PSNR: %f, SSIM: %f\n" %
          ( psnrs.avg, ssims.avg))