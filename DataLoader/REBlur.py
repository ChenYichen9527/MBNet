import h5py
from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
from myutils import *
import matplotlib.pyplot as plt
class reblur(Dataset):
    def __init__(self, data_root, is_training):
        self.data_root = data_root
        
        self.training = is_training
        self.paths=[]
        for s in os.listdir(self.data_root):
            self.paths.append (os.path.join(self.data_root , s) )
            
        self.file_index=[]
        self.img_index=[]
        for i in range(len(self.paths)):
           
            h5_file = self.paths[i]
            self.h5_file = h5py.File( h5_file ,'a')
            self.frame_ts = []
            for img_name in self.h5_file['images']:
                self.frame_ts.append(self.h5_file['images/{}'.format(img_name)].attrs['timestamp'])

            for k in range(1,len(self.frame_ts)-1): 
                # self.frame_ts.append(self.h5_file['images/{}'.format(img_name)].attrs['timestamp'])
                self.file_index.append(i)
                self.img_index.append(k-1)

       
        
    def get_event_indices(self, index):
        """
        Get start and end indices of events at index
        @param Desired data index
        @returns Start and end indices of events at index
        """
        idx0, idx1 = self.event_indices[index]
        if not (idx0 >= 0 and idx1 <= self.num_events):
            raise Exception("WARNING: Event indices {},{} out of bounds 0,{}".format(idx0, idx1, self.num_events))
        return int(idx0), int(idx1)
    def preprocess_events(self,xs, ys, ts, ps):
        """
        Given empty events, return single zero event
        @param xs x compnent of events
        @param ys y compnent of events
        @param ts t compnent of events
        @param ps p compnent of events
        """
        if len(xs) == 0:
            txs = np.zeros((1))
            tys = np.zeros((1))
            tts = np.zeros((1))
            tps = np.zeros((1))
            return txs, tys, tts, tps
        return xs, ys, ts, ps

    def compute_frame_center_indeices(self):
        """
        For each frame, find the start and end indices of the events around the
        frame, the start and the end are at the middle between the frame and the 
        neighborhood frames
        """
        frame_indices = []
        start_idx = self.find_ts_index((self.frame_ts[0]+self.frame_ts[1])/2)
        for i in range(1, len(self.frame_ts)-1): 
            end_idx = self.find_ts_index((self.frame_ts[i]+self.frame_ts[i+1])/2)
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices

    def get_frame(self, index):
        """
        discard the first and the last frame in GoproEsim
        """
        return self.h5_file['images']['image{:09d}'.format(index+1)][:]

    def get_gt_frame(self, index):
        """
        discard the first and the last frame in GoproEsim
        """
        return self.h5_file['sharp_images']['image{:09d}'.format(index+1)][:]

    def get_events(self, idx0, idx1):
        xs = self.h5_file['events/xs'][idx0:idx1]
        ys = self.h5_file['events/ys'][idx0:idx1]
        ts = self.h5_file['events/ts'][idx0:idx1]
        ps = self.h5_file['events/ps'][idx0:idx1] * 2.0 - 1.0  # -1 and 1
        return xs, ys, ts, ps
    def transform_frame(self, frame, seed, transpose_to_CHW=False):
    
        frame = torch.from_numpy(frame).float().unsqueeze(0) / 255
        # if self.transform:
        #     random.seed(seed)
        #     print('frame.shape:{}'.format(frame.shape))
        #     frame = self.transform(frame)
        return frame




    def binary_search_h5_dset(self,dset, x, l=None, r=None, side='left'):

        l = 0 if l is None else l
        r = len(dset)-1 if r is None else r
        while l <= r:
            mid = l + (r - l)//2;
            midval = dset[mid]
            if midval == x:
                return mid
            elif midval < x:
                l = mid + 1
            else:
                r = mid - 1
        if side == 'left':
            return l
        return r

    def find_ts_index(self, timestamp):
        idx = self.binary_search_h5_dset(self.h5_file['events/ts'], timestamp)
        return idx

    def ts(self, index):
        return self.h5_file['events/ts'][index]

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for img_name in self.h5_file['images']:
            end_idx = self.h5_file['images/{}'.format(img_name)].attrs['event_idx']
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices
    def __getitem__(self, index):
        file_i=self.file_index[index]
        img_i =self.img_index[index]
        
        h5_file=self.paths[file_i]
        self.sensor_resolution, self.t0, self.tk, self.num_events, self.frame_ts, self.num_frames = \
                None, None, None, None, None, None
        seed = random.randint(0, 2 ** 32) 
        self.data_sources = ('esim', 'ijrr', 'mvsec', 'eccd', 'hqfd', 'unknown')
        
        self.h5_file = h5py.File( h5_file ,'a')

        if self.sensor_resolution is None:
            self.sensor_resolution = self.h5_file.attrs['sensor_resolution'][0:2]
        else:
            self.sensor_resolution = self.sensor_resolution[0:2]
        h,w =self.sensor_resolution
        # print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['flow']) > 0
        self.t0 = self.h5_file['events/ts'][0]
        self.tk = self.h5_file['events/ts'][-1]
        self.num_events = self.h5_file.attrs["num_events"]
        self.num_frames = self.h5_file.attrs["num_imgs"]
        self.length = self.num_frames - 2
        self.frame_ts = []
        for img_name in self.h5_file['images']:
            self.frame_ts.append(self.h5_file['images/{}'.format(img_name)].attrs['timestamp'])

        data_source = self.h5_file.attrs.get('source', 'unknown')
        try:
            self.data_source_idx = self.data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1
        self.event_indices = self.compute_frame_center_indeices()
        k=img_i
            
        idx0, idx1 = self.get_event_indices(k) # the start and end index of the selected events
        # print('DEBUG: idx0:{}, idx1:{}'.format(idx0, idx1))
        xs, ys, ts, ps = self.get_events(idx0, idx1) # the selected events, determined by the voxel method
        xs, ys, ts, ps = self.preprocess_events(xs, ys, ts, ps)
        ts_0, ts_k  = ts[0], ts[-1]
        # events = np.stack([xs, ys, ts, ps ])
        dt = ts_k-ts_0
        frame = self.get_frame(k)
        frame_gt = self.get_gt_frame(k)
        # frame = self.transform_frame(frame, seed, transpose_to_CHW=True) # to tensor
        # frame_gt = self.transform_frame(frame_gt, seed, transpose_to_CHW=True)


        if ts.shape[0] > 1:
            # timestamp normalization
      
            ts = (ts - ts.min()) / (ts.max() - ts.min() + 1e-5) 

        eventpoints = np.stack([ ts, xs, ys, ps], 1)
        eventpoints = torch.Tensor( eventpoints)


        left_events,right_events,mask=e_split(eventpoints,0.5)
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
      
        img2 = torch.Tensor(frame.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(frame_gt.copy()).float().permute(2, 0, 1) / 255.0

        return img2, img2, img2,img_gt, left_reverse_vol,left_vol,right_vol


   




        return item
    def __len__(self):
        return len(self.file_index)
if __name__ == "__main__":
    root=r'/media/root/f/LongguangWang/Data/REBlur_rawevents/test'
    data = reblur( root,is_training=False)
    dataloader = DataLoader(dataset=data, batch_size=1,shuffle=False)

    _iter = 0
    for i in dataloader:
        event_image = i[0]
        print(_iter, event_image.shape)