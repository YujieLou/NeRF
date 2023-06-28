import os, sys
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
# device_ids = [0, 1, 2, 3]
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random
import time
import imageio
import numpy as np
from tqdm import tqdm, trange
import configargparse

import torch
import torch.nn as nn
import tinycudann as tcnn

from utils import *
from models import NeRF, Embedder
from dataloader import load_llff_data

seed_everything(0)



    # loaddata里面的是render poses是

    
    if args.render_test:
        render_poses = np.array(poses[i_test])