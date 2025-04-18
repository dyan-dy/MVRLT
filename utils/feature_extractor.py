# extract pretrained or self-supervised features for enhancing multiview information
# type: DINO, CLIP(for local editing, color and material perception, ..), optical-flow (dense view), depth, distilled 3dgs or nerf
# TODO

import os
import copy
import sys
import json
import importlib
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from torchvision import transforms
from PIL import Image

# load model
dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
dinov2_model.eval().cuda()
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
n_patch = 518 // 14

