from sqlite3 import Timestamp
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from helper_tool import Plot
from tqdm import tqdm
import myutils
from loss import Loss
from DataLoader.mydataloader import Goprosataset

from torch.optim import Adam
from model import myNet
import argparse
import torch.nn.functional as F




def Goprotest(epoch, mymodel, criterion, testloader,device,args):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims = myutils.init_meters()
    mymodel.eval()
    criterion.eval()
        
    with torch.no_grad():
        for i, (img1, img2,img3, img_gt, left_reverse_vol,left_vol,right_vol) in enumerate(tqdm(testloader)):
            img1 = img1.float().to(device)        #([b, 2, 3, 180, 320])
            img2 = img2.float().to(device)        #([b, 2, 3, 180, 320])
            img3 = img3.float().to(device)
            img_gt = img_gt.float().to(device)  #([b, 3, 180, 320])
            left_reverse_vol = left_reverse_vol.float().to(device)    #[1, 200000, 4])
            left_vol = left_vol.float().to(device)  
            right_vol = right_vol.float().to(device)  

            img_out= mymodel(img1, img2, img3,left_reverse_vol,left_vol,right_vol,args)
            # Evaluate metrics
            
            myutils.eval_metrics(img_out, img_gt, psnrs, ssims)
            myutils.save_testing(img_out, img_gt ,i)        
    # Print progress
    print(" PSNR: %f, SSIM: %f\n" %
          ( psnrs.avg, ssims.avg))

    return losses.avg, psnrs.avg, ssims.avg




def vimeotest(epoch, mymodel, criterion, testloader,device,testdataset,args):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims = myutils.init_meters()
    mymodel.eval()
    criterion.eval()
    allloss=0 
    with torch.no_grad():
    
        for i, (img1, img2,img3, img_gt, left_reverse_vol,left_vol,right_vol) in enumerate(tqdm(testloader)):

                img1 = img1.float().to(device)        #([b, 2, 3, 180, 320])
                img2 = img2.float().to(device)        #([b, 2, 3, 180, 320])
                img3 = img3.float().to(device)
                img_gt = img_gt.float().to(device)  #([b, 3, 180, 320])
                left_reverse_vol = left_reverse_vol.float().to(device)    #[1, 200000, 4])
                left_vol = left_vol.float().to(device)  
                right_vol = right_vol.float().to(device)  

                s= 1
                img_out= mymodel(img1, img2, img3,left_reverse_vol,left_vol,right_vol,s,args)
                img_out = torch.clamp(img_out,0,1)
                # img_gt = F.upsample(img_gt,[int(img_out.shape[2]),int(img_out.shape[3])],mode='bilinear')
                myutils.eval_metrics(img_out, img_gt, psnrs, ssims)
            
                myutils.save_testing(img_out, img_gt ,i)        
    # Print progress
    print(" PSNR: %f, SSIM: %f\n" %
          ( psnrs.avg, ssims.avg))

    return losses.avg, psnrs.avg, ssims.avg

