# from https://github.com/myungsub/CAIN/blob/master/utils.py, 
# but removed the errenous normalization and quantization steps from computing the PSNR.

from multiprocessing import Event
from turtle import right
from numpy import linspace
from pytorch_msssim import ssim_matlab as calc_ssim
import math
import os
import torch
import shutil

import matplotlib.pyplot as plt
import copy
from helper_tool import Plot
from torchvision import transforms
import cv2 as cv
import numpy as np

def init_meters():
    losses = AverageMeter()
    psnrs = AverageMeter()
    ssims = AverageMeter()
    return losses, psnrs, ssims

def eval_metrics(output, gt, psnrs, ssims):
    # PSNR should be calculated for each image, since sum(log) =/= log(sum).
    for b in range(gt.size(0)):
        psnr = calc_psnr(output[b], gt[b])
        psnrs.update(psnr)

        ssim = calc_ssim(output[b].unsqueeze(0).clamp(0,1), gt[b].unsqueeze(0).clamp(0,1) , val_range=1.)
        ssims.update(ssim)

# def init_losses(loss_str):
#     loss_specifics = {}
#     _, loss_type = loss_str.split('*')
#     loss_specifics[loss_type] = AverageMeter()
#     loss_specifics['total'] = AverageMeter()
#     return loss_specifics


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_psnr(pred, gt):
    diff = (pred - gt).pow(2).mean() + 1e-8
    return -10 * math.log10(diff)


def save_checkpoint(state, directory, is_best,  filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, 'model_best.pth'))

def log_tensorboard(writer, loss, psnr, ssim, lpips, lr, timestep, mode='train'):
    writer.add_scalar('Loss/%s/%s' % mode, loss, timestep)
    writer.add_scalar('PSNR/%s' % mode, psnr, timestep)
    writer.add_scalar('SSIM/%s' % mode, ssim, timestep)
    if mode == 'train':
        writer.add_scalar('lr', lr, timestep)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_mid_results(img_out,imgs,gt_img,events,epoch,i,loss):
    show_img=img_out[0,0,...].data.cpu()
    show_o1=imgs[0,0,0,...].data.cpu()
    show_o2=imgs[0,1,0,...].data.cpu()
    show_gt=gt_img[0,0,...].cpu()
    show_events=events[0,...].cpu()
    show_evn=checkdisplay2(show_o1,show_o2, show_events)
    plt.figure()
    
    plt.imshow(show_evn)
    plt.savefig('./pics/resultsfile/pic'+str(i)+'epoch'+str(epoch+1)+'loss'+str(loss)+'event.png')

    plt.imshow(show_img)
    plt.savefig('./pics/resultsfile/pic'+str(i)+'epoch'+str(epoch+1)+'loss'+str(loss)+'out.png')
    # plt.show()
    plt.imshow(show_gt)
    plt.savefig('./pics/resultsfile/pic'+str(i)+'epoch'+str(epoch+1)+'loss'+str(loss)+'o2gt.png')

    plt.imshow(show_o1)
    plt.savefig('./pics/resultsfile/pic'+str(i)+'epoch'+str(epoch+1)+'loss'+str(loss)+'o1.png')
    plt.imshow(show_o2)
    plt.savefig('./pics/resultsfile/pic'+str(i)+'epoch'+str(epoch+1)+'loss'+str(loss)+'o3.png')



# def save_testing(img_out, img_gt ):


def display_neighbor(neighbor_idx,xyt,color=None):
    """
    neighbor_idx: the neighbor index
    xyt: the coordinates of the points
    plot_colors: custom color list
    """

    predictions=np.int64(np.zeros(xyt[:,-1].shape))
    for points_i in range(xyt.shape[0]):
        this_prediction=copy.deepcopy(predictions)
 
        this_prediction[ neighbor_idx[points_i]] = 1
        this_prediction[points_i]=2

        if color == None:
            Plot.draw_pc_sem_ins(xyt, this_prediction)
        else:
            Plot.draw_pc_sem_ins(xyt, this_prediction,color)


def To_norm(flow):
    if flow.max()-flow.min()!=0:
        image_optical_flow = (flow-flow.min())/(flow.max()-flow.min())*2-1
    else:
        image_optical_flow = flow
        
    return image_optical_flow


def display_evn_img(xytp,Im1,Im2):
    xytp=torch.cat((xytp[:,0:3]*47,(xytp[:,-1]+1).unsqueeze(-1)/2),dim=1)
    Event=torch.cat( (xytp,torch.zeros((xytp.shape[0],2)).to(xytp.device)  )  ,dim=1)
    # Event=torch.cat((xytpp,torch.zeros((xytp.shape[0],1)) ),dim=1)
    pos_ind = Event[:,3]==255
    Event_pos=Event[pos_ind]

    neg_ind = Event[:,3]==0
    Event_neg=Event[neg_ind]
    

    _,h,w=Im1.shape
    Im1=Im1.permute(1,2,0)
    Im2=Im2.permute(1,2,0)

    index=torch.linspace(0,h*w-1,h*w).to(xytp.device)
    y = (index/h).long()

    x =(index % h).long()
    Im1=torch.tensor(Im1.reshape(-1,3))
    Im2=torch.tensor(Im2.reshape(-1,3))
    xy=torch.cat((x.unsqueeze(1),y.unsqueeze(1)),dim=1)

    Im1xyt=torch.cat([torch.zeros((xy.shape[0],1)).to(xytp.device),xy,Im1],dim=1)
    Plot.draw_pc(Im1xyt.data.cpu())

    Im2xyt=torch.cat([h*torch.ones((xy.shape[0],1)).to(xytp.device),xy,Im2],dim=1)
    # Im2xyt=torch.cat(,Im2xy),dim=1)
  
    allim=torch.vstack((Im2xyt,Im1xyt))
    all=torch.vstack((allim,Event))
    allpos=torch.vstack((allim,Event_pos))
    allneg=torch.vstack((allim,Event_neg))
    Plot.draw_pc(all.data.cpu())
    # Plot.draw_pc(allneg.data.cpu())
    # Plot.draw_pc(allpos.data.cpu())


def events_split(evn_fea, events, t_stamp):
    b, c, n =evn_fea.shape
    left_fea  = []
    right_fea = []
    mask = []
    for bi in range(b):
        t = events[bi, :, 0]
        left_ind = t<t_stamp[bi]
        right_ind = ~left_ind
        left_fea.append(evn_fea[bi, :, left_ind])
        right_fea.append(evn_fea[bi,:, right_ind])
        mask.append(left_ind)
    mask = torch.stack(mask)


    return left_fea, right_fea, mask
def e2v(fea, events, e_ratio,img_ratio, mask, bins=10, c=32):
    b,_,h,w = img_ratio.shape
    layers=[]
   
    for bi in range(b):
        
        img_fea = torch.zeros(c, bins * h * w).to(events.device)
        img_num = torch.ones(1,  bins * h * w).to(events.device)
        

        maski = mask[bi]
        feai = fea[bi]
        xyt = events[bi, maski, :3]
        t_ratio  = e_ratio[bi,:, maski].reshape(-1)
        im_ratio = img_ratio[bi, ...].reshape(-1)
        
        if xyt.shape[0] == 0:
            layers.append(img_fea.reshape(c,bins,h,w))
            continue

        t_ind =(xyt[:,0] - xyt[:,0].min()) / (xyt[:,0].max() - xyt[:,0].min() + 1e-5) 
        t_ind = torch.clamp( (t_ind * bins).long(), 0, bins-1)

        x_ind = (xyt[:, 1] * w ).long()
        y_ind = (xyt[:, 2] * h ).long()

        # ind = (bins * h * t_ind + h * y_ind + x_ind)  

        ind = (w * h * t_ind + h * y_ind + x_ind) 

        s_ratio = im_ratio[ h * y_ind + x_ind]

        the_fea = t_ratio * s_ratio * feai

        img_fea.index_add_( 1, ind, the_fea.reshape(c,-1) )
        img_num.index_add_( 1, ind, t_ratio.reshape(1,-1) * 10 )

        final = 10*img_fea / img_num
        
        layers.append(final.reshape(c,bins,h,w))

    return torch.stack(layers)


# def save_testing(img_out, img_gt,pic_i ) :
#     b,c,h,w = img_out.shape
#     imgs=[]
#     for index in range(b):
#         imgs = transforms.ToPILImage()(img_out[index]) 
#         imgs_gt = transforms.ToPILImage()(img_gt[index]) 
#         imgs.save("pics/1/{:04d}".format(pic_i)+str(index)+"0.png")
#         imgs_gt.save("pics/1/{:04d}".format(pic_i)+str(index)+"1.png")




def checkdisplay(Im1,Im2,eventpoints,is_show=False):

    Im1=np.float32(Im1)
    Im2=np.float32(Im2)
  
    Imtest=np.abs(Im1-Im2)
    Imtest_copy=np.uint8(Imtest)
    Imtest=np.uint8(Imtest)
    for line in eventpoints:
        if line[3]==-1: 
                #负
            Imtest[int(line[2]),int(line[1])]=[255,0,0]
        else:
            Imtest[int(line[2]),int(line[1])]=[0,0,255]
    theIm=np.concatenate((Imtest,Imtest_copy),axis=1)

    savename='pic.png'
    cv.imwrite(savename, theIm) 
    if is_show:
        # cv.imwrite(os.path.join(savepath,savename), theIm) 
        cv.namedWindow("this", 0)
        cv.resizeWindow("this",  1280,640)  # 设置窗口大小
        cv.imshow("this", theIm)
        cv.waitKey(0)


def checkdisplay2(Im1,Im2,eventpoints,is_save=False,is_show=True):

    Im1=np.float32(Im1)
    Im2=np.float32(Im2)
    h,w,c =Im1.shape
    Imtest=np.abs(Im1-Im2)
    Imtest_copy=np.uint8(Imtest)
    Imtest=np.uint8(Imtest)
    for line in eventpoints:
        if line[3]==-1: 
                #负
            Imtest[int(line[2]*h),int(line[1]*w)]=[255,0,0]
        else:
            Imtest[int(line[2]*h),int(line[1]*w)]=[0,0,255]
    theIm=np.concatenate((Imtest,Imtest_copy),axis=1)
    if is_save:
        savename='pic.png'
        cv.imwrite(savename, theIm) 
    if is_show:
        # cv.imwrite(os.path.join(savepath,savename), theIm) 
        cv.namedWindow("this", 0)
        cv.resizeWindow("this",  1280,640)  # 设置窗口大小
        cv.imshow("this", theIm)
        cv.waitKey(0)


def e_split(events, t_stamp):

    t = events[:, 0]
    left_ind = t<t_stamp
    right_ind = ~left_ind
    left_fea = (events[left_ind,:])
    right_fea = (events[right_ind, :])
    mask = (left_ind)


    return left_fea, right_fea, mask



def e_reverse(events):
    """Reverse temporal direction of the event stream.

    Polarities of the events reversed.

                        (-)       (+)
    --------|----------|---------|------------|----> time
        t_start        t_1       t_2        t_end

                        (+)       (-)
    --------|----------|---------|------------|----> time
            0    (t_end-t_2) (t_end-t_1) (t_end-t_start)

    """
    events_copy=events.clone()
    if len(events) == 0:
        return events
    events_copy[:, 0] = (events_copy[:, 0].max() - events_copy[:, 0])
    events_copy[:, -1] = -events_copy[:, -1]

    # Flip rows of the 'features' matrix, since it is sorted in oldest first.
    events_copy = np.copy(np.flipud(events_copy))
    return events_copy


def to_voxel_grid(event_sequence, nb_of_time_bins=8,h=48,w=48,remapping_maps=None):
    """Returns voxel grid representation of event steam.

    In voxel grid representation, temporal dimension is
    discretized into "nb_of_time_bins" bins. The events fir
    polarities are interpolated between two near-by bins
    using bilinear interpolation and summed up.

    If event stream is empty, voxel grid will be empty.
    """

    voxel_grid = torch.zeros(nb_of_time_bins,h,w,dtype=torch.float32,device='cpu')

    if event_sequence.shape[0]==0:
        return voxel_grid

    voxel_grid_flat = voxel_grid.flatten()

    # Convert timestamps to [0, nb_of_time_bins] range.
    duration = event_sequence[:,0].max()-event_sequence[:,0].min()
    start_timestamp = event_sequence[:,0].min()
    features = torch.from_numpy(event_sequence)
    x = features[:, 1]
    y = features[:, 2]
    polarity = features[:, -1].float()
    t = (features[:, 0] - start_timestamp) * (nb_of_time_bins - 1) / duration
    t = t.float()

    if remapping_maps is not None:
        remapping_maps = torch.from_numpy(remapping_maps)
        x, y = remapping_maps[:,y,x]

    left_t, right_t = t.floor(), t.floor()+1
    left_x, right_x = x.floor(), x.floor()+1
    left_y, right_y = y.floor(), y.floor()+1

    for lim_x in [left_x, right_x]:
        for lim_y in [left_y, right_y]:
            for lim_t in [left_t, right_t]:
                mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= w-1) \
                       & (lim_y <= h-1) & (lim_t <= nb_of_time_bins-1)

                # we cast to long here otherwise the mask is not computed correctly
                lin_idx = lim_x.long() + lim_y.long() * w + lim_t.long() * w * h
                weight = polarity * (1-(lim_x-x).abs()) * (1-(lim_y-y).abs()) * (1-(lim_t-t).abs())
                voxel_grid_flat.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())
    # print(voxel_grid.max())
    return voxel_grid




def to_mask(event_sequence, nb_of_time_bins=1,h=48,w=48,remapping_maps=None):


    voxel_grid = torch.zeros(h,w,dtype=torch.float32,device='cpu')

    if event_sequence.shape[0]==0:
        return voxel_grid

    voxel_grid_flat = voxel_grid.flatten()


    features =event_sequence
    x = features[:, 1]
    y = features[:, 2]
    polarity = features[:, -1].float()
    p2=torch.ones(polarity.shape)



    left_x, right_x = x.floor(), x.floor()+1
    left_y, right_y = y.floor(), y.floor()+1

    for lim_x in [left_x, right_x]:
        for lim_y in [left_y, right_y]:
            mask = (0 <= lim_x) & (0 <= lim_y)  & (lim_x <= w-1)   & (lim_y <= h-1) 
            lin_idx = lim_x.long() + lim_y.long() * w 
            voxel_grid_flat.index_add_(dim=0, index=lin_idx[mask], source=p2[mask].float())

    return voxel_grid



def save_testing(img_out, img_gt,pic_i ) :
    b,c,h,w = img_out.shape
    imgs=[]
    for index in range(b):
        imgs = transforms.ToPILImage()(img_out[index]) 
        imgs_gt = transforms.ToPILImage()(img_gt[index]) 
        savepath= r"/media/root/LENOVO_USB_HDD/Result/Deblur_Results/result/code_dn_v2/REBlur"
        imgs.save(savepath+"/{:08d}".format(pic_i)+str(index)+"0.png")
        imgs_gt.save(savepath+"/{:08d}".format(pic_i)+str(index)+"1.png")




