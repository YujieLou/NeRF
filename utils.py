import os
import random 
import logging
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

import time 




# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

def rotate_matrix(yaw, pitch, roll):
    R_z = torch.zeros([3, 3])
    R_z[0][0], R_z[0][1], R_z[1][0], R_z[1][1], R_z[2][2] = np.cos(yaw), -np.sin(yaw), np.sin(yaw), np.cos(yaw), 1
    R_y = torch.zeros([3, 3])
    R_y[0][0], R_y[0][2], R_y[2][0], R_y[2][2], R_y[1][1] = np.cos(pitch), np.sin(pitch), -np.sin(pitch), np.cos(pitch), 1
    R_x = torch.zeros([3, 3])
    R_x[1][1], R_x[1][2], R_x[2][1], R_x[2][2], R_x[0][0] = np.cos(roll), -np.sin(roll), np.sin(roll), np.cos(roll), 1
    R = torch.mm(torch.mm(R_z, R_y), R_x)
    print("ADD Rotate", R)
    return R


def isRayIntersectsSegment(pt, s_pt, e_pt):
    if s_pt[1] == e_pt[1]:
        return False
    if s_pt[1] > pt[1] and e_pt[1] > pt[1]:
        return False
    if s_pt[1] < pt[1] and e_pt[1] < pt[1]:
        return False
    if s_pt[1] == pt[1] and e_pt[1] > pt[1]:
        return False  
    if e_pt[1] == pt[1] and s_pt[1] > pt[1]:
        return False  
    if s_pt[0] < pt[0] and e_pt[0] < pt[0]:
        return False
    
    xseg = e_pt[0] - (e_pt[0]-s_pt[0])*(e_pt[1]-pt[1])/(e_pt[1]-s_pt[1])
    if xseg < pt[0]:
        return False
    return True
    
    
def isPointWithinPolygon(pt, poly):
    sinsc = 0
    n = len(poly)
    for i in range(n-1):
        s_pt = poly[i]
        e_pt = poly[i+1]
        if isRayIntersectsSegment(pt, s_pt, e_pt):
            sinsc += 1
    s_pt = poly[-1]
    e_pt = poly[0]
    if isRayIntersectsSegment(pt, s_pt, e_pt):
        sinsc += 1
    
    return True if sinsc%2==1 else False


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    return rays_o, rays_d


def get_rays(H, W, K, c2w):
    if torch.is_tensor(c2w):
        i, j = torch.meshgrid(torch.linspace(0, W-1, W).to(c2w.device), torch.linspace(0, H-1, H).to(c2w.device))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3,-1].expand(rays_d.shape)
    else:
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
        rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def get_rays_np_choice(H, W, K, c2w, car_bound):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    rays_o.flags.writeable = True
    rays_d.flags.writeable = True
    tag = 0
    print('image size', H, W)
    for bd in car_bound:
        x_max, x_min = int(max([item[0] for item in bd])), int(min([item[0] for item in bd]))
        y_max, y_min = int(max([item[1] for item in bd])), int(min([item[1] for item in bd]))
        for i in range(x_min, x_max+1):
            for j in range(y_min, y_max+1):
                if isPointWithinPolygon([i,j], bd):
                    rays_d[j][i] = [-1000, -1000, -1000]
                    tag +=1
    print(tag , '/' , H * W)
    return rays_o, rays_d


def remove_car(rays_rgb):
    idxs = []
    for _ in range(len(rays_rgb)):
        if not -1<(rays_rgb[_][1][0] + 1000) < 1:
            idxs.append(_)
    print('see num difference',len(rays_rgb), len(idxs))
    rays_rgb = rays_rgb[idxs]
    return rays_rgb
    

def get_logger(filename, distributed_rank=0, gpus=False, verbosity=1, name=None):
    if gpus:
        if distributed_rank > 0:
            logging_not_root = logging.getLogger(name=__name__)
            logging_not_root.propagate = False
            return logging_not_root
    
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger