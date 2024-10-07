from pickle import TRUE
import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from helper_tool import Plot
from tqdm import tqdm
import myutils
from loss import Loss
from DataLoader.mydataloader_sf import Goprosataset_sf
from DataLoader.fullgoprotestloaders_sf import Goprotestdataset_sf
from DataLoader.mydataloader_hr import Goprosataset_hr
from DataLoader.fullgoprotestloaders_hr import Goprotestdataset_hr,Goprotestdataset_hr_deblur
from DataLoader.HQFtest_hr import HQFset_hr
from DataLoader.HQFtest import HQFset
from DataLoader.REBlur import reblur
from DataLoader.REBlurtrain import reblur_train
from torch.optim import Adam
from model import myNet
import argparse
from training import train,Vimeotrain
from testing import *


parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=bool, default=True, help="use cuda or not")
parser.add_argument("--load_checkpoints", type=bool, default=True, help="...")

parser.add_argument("--train_batchsize", type=int, default=4, help="the batchsize setting when training")
parser.add_argument("--test_batchsize", type=int, default=1, help="the batchsize setting when testing")

parser.add_argument("--epochs", type=int, default=30, help="the total epochs")
parser.add_argument("--init_epoch", type=int, default=0, help="the initial epoch")

parser.add_argument("--workers", type=int, default=2, help=" ")
parser.add_argument("--sequences", type=int, default=1, help="the initial epoch")
args = parser.parse_args()



if __name__ == "__main__":

    init_epoch = args.init_epoch
  
    data_path = r"/media/root/f/LongguangWang/Data/Goprodataset"

    # traindataset = Goprosataset_sf(data_path, is_training=True)
    # trainloader = DataLoader(traindataset, batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)

  
    # testdataset = Goprosataset(data_path, is_training=False)
    # testloader = DataLoader(testdataset, batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

  
    # testdataset =Goprotestdataset_sf(data_path,is_training=False)
    # # testdataset = Goprotestdataset_hr_deblur(data_path,is_training=False)
    # testloader = DataLoader(testdataset, batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)



    # testdataset= HQFset(r'/media/root/LENOVO_USB_HDD/HQF', is_training=False) 
    # testloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    traindataset= reblur_train(r'/media/root/f/LongguangWang/Data/REBlur_rawevents/train',is_training=True)
    trainloader = DataLoader(traindataset, batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
    
      
    testdataset= reblur(r'/media/root/f/LongguangWang/Data/REBlur_rawevents/test',is_training=False)
    testloader = DataLoader(dataset=testdataset, batch_size=1,shuffle=False)
    
    criterion = Loss()

    device = torch.device('cuda:0' if args.cuda else 'cpu')
    
    mymodel= myNet(args.sequences*3).to(device)
    if args.cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        mymodel = torch.nn.DataParallel(mymodel).to(device)


    total = sum([param.nelement() for param in mymodel.parameters()])

    print("Number of parameter: %.2fM" % (total/1e6))
    # num = 500000
    # len_neighbors  = [ num, num//4,  num//16, num//4,num]
    
    # num_neighbors =  [8,8,8,1,1]
    # neighbors = []
    # for ki, l in enumerate( len_neighbors):
    #     neighbors.append( torch.zeros([1,l,num_neighbors[ki]], dtype=torch.int64).cuda())

    # events_input =torch.zeros([1,num, 4]).cuda()
    # dummy_input1 = torch.randn(1, 3, 224, 224).cuda()
    # dummy_input2 = torch.randn(1, 3, 224, 224).cuda()
    # flops, params = profile(mymodel.cuda(), (events_input,dummy_input1,dummy_input2,neighbors,[0.5]))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))


    mymodel.train()
    optimizer = Adam(mymodel.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    best_psnr = 0

    if args.load_checkpoints:
        model_path = 'logs/ep018-psnr38.195-ssim0.9756-loss0.1011.pth'
        checkpoint = torch.load(model_path, map_location = device)
        mymodel.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # init_epoch = checkpoint['epoch']



    for epoch in range(init_epoch, args.epochs):

        # losses = Vimeotrain(trainloader, epoch,  mymodel, criterion, optimizer,traindataset,args)
        # lr_scheduler.step()
        # if epoch%3 !=2:
        #         continue
        test_loss, psnr, ssim = vimeotest(epoch, mymodel, criterion, testloader,device,testdataset,args)
        # test_loss, psnr, ssim = vimeotest(epoch, mymodel, criterion, testloader_2,device,testdataset_2,args)
        
            
        # save checkpoint
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)

        myutils.save_checkpoint({
            'epoch': epoch,
            'state_dict': mymodel.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_psnr,
            'lr': optimizer.param_groups[-1]['lr']
        }, './logs', is_best, 'ep%03d-psnr%.3f-ssim%.4f-loss%.4f.pth' % (epoch,psnr,ssim,losses))
        lr_scheduler.step()
        # if  epoch==20:
        #     test_loss, psnr, ssim = vimeotest(epoch, mymodel, criterion, testloader_2,device,testdataset_2,args)
    # test_loss, psnr, ssim = vimeotest(epoch, mymodel, criterion, testloader_small,device,args)