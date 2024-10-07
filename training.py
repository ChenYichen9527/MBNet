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
import torch.nn.functional as F
from torch.optim import Adam
from model import myNet
import argparse
import  numpy as np


def train(trainloader, epoch, mymodel, criterion, optimizer,traindataset,args):
    mymodel.train()
    criterion.train()
    losses =0
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    with tqdm(total=len(traindataset)//args.train_batchsize, desc=f'Epoch {epoch + 1}/{args.epochs}',postfix=dict,mininterval=0.3) as pbar:
        for i, (img1, img2, img_gt, events, neighbors) in enumerate(trainloader):
            optimizer.zero_grad()

            img1 = img1.to(device)        #([b, 2, 3, 180, 320])
            img2 = img2.to(device)        #([b, 2, 3, 180, 320])
            img_gt = img_gt.float().to(device)  #([b, 3, 180, 320])
            events = events.to(device)    #[1, 200000, 4])
            neighbors = [x.to(device) for x in neighbors]

            img_out,img_out_dn = mymodel(events, img1, img2, neighbors,1)[0]

            loss_sr = criterion(img_out, img_gt)
            # loss = loss_sr + 0.1 * loss_flow
            loss = loss_sr 
            losses += loss.item()

            loss.backward()
            optimizer.step()
            # pbar.set_postfix(**{'loss_sr': loss_sr.item(), 'loss_flow': loss_flow.item(), 'lr':  myutils.get_lr(optimizer)})
            pbar.set_postfix(**{'loss_sr': losses/(i+1),  'lr':  myutils.get_lr(optimizer)})
            pbar.update(1)




def Vimeotrain(trainloader, epoch, mymodel, criterion, optimizer,traindataset,args):
    mymodel.train()
    criterion.train()
    losses =0
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    with tqdm(total=len(traindataset)//args.train_batchsize, desc=f'Epoch {epoch + 1}/{args.epochs}',postfix=dict,mininterval=0.3) as pbar:
        for i, (img1, img2,img3, img_gt, left_reverse_vol,left_vol,right_vol) in enumerate(trainloader):
            optimizer.zero_grad()

            img1 = img1.float().to(device)        #([b, 2, 3, 180, 320])
            img2 = img2.float().to(device)        #([b, 2, 3, 180, 320])
            img3 = img3.float().to(device)
            img_gt = img_gt.float().to(device)  #([b, 3, 180, 320])
            left_reverse_vol = left_reverse_vol.float().to(device)    #[1, 200000, 4])
            left_vol = left_vol.float().to(device)  
            right_vol = right_vol.float().to(device)  


  
            s = 1        
     
            img_out= mymodel(img1, img2, img3,left_reverse_vol,left_vol,right_vol,s,args)
            
            label_fft = torch.fft.fft2(img_gt, dim=(-2, -1))
            label_fft = torch.stack((label_fft.real, label_fft.imag), -1)
            pred_fft = torch.fft.fft2(img_out, dim=(-2, -1))
            pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)



            # myutils.save_testing(img_out, img_gt ,i)    
            loss_sr= criterion(img_out, img_gt)
            loss_fft = criterion(pred_fft, label_fft)
       
            loss = loss_sr + 0.1*loss_fft
            
            # loss =loss_sr
            
            losses += loss.item()
            loss.backward()
            optimizer.step()
            # losses=0
            # pbar.set_postfix(**{'loss_sr': loss_sr.item(), 'loss_flow': loss_flow.item(), 'lr':  myutils.get_lr(optimizer)})
            pbar.set_postfix(**{'loss_sr': losses/(i+1),  'lr':  myutils.get_lr(optimizer)})
            pbar.update(1)
    return losses/(i+1)