import open3d as open3d
from os.path import join
import numpy as np
import colorsys, random, os, sys
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))



class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb, window_name='Open3D'):
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            open3d.visualization.draw_geometries([pc])
            return 0
        if pc_xyzrgb[:, 3:6].max() > 20:  ## 0-255
            pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
        open3d.visualization.draw_geometries([pc], window_name=window_name)
        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None, window_name='labels'):
        """
        pc_xyz: 3D coordinates of point clouds
        pc_sem_ins: semantic or instance labels
        plot_colors: custom color list
        """
        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=2)

        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)   #只有1
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))

        for id, semins in enumerate(sem_ins_labels):
            valid_ind=np.where(pc_sem_ins == semins)
      

            #找到有效ind
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            ### bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = valid_xyz[:, 0].min()#np.min(valid_xyz[:, 0])
            xmax = valid_xyz[:, 0].max()#np.max(valid_xyz[:, 0])
            ymin = valid_xyz[:, 1].min()#np.min(valid_xyz[:, 1])
            ymax = valid_xyz[:, 1].max()#np.max(valid_xyz[:, 1])
            zmin = valid_xyz[:, 2].min()#np.min(valid_xyz[:, 2])
            zmax = valid_xyz[:, 2].max()#np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins, window_name=window_name)
        return Y_semins
