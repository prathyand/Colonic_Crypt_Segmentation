import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import monai
from monai.networks.nets import unetr
from monai.losses import DiceLoss, DiceCELoss


class modelconf(object):

    def __init__(self,MODEL,LOSS_F=None,Device=None,Optim=None):
        self.MODEL=MODEL
        self.LOSS_F = LOSS_F if LOSS_F else DiceCELoss(to_onehot_y=False, sigmoid=True)
        self.DEVICE = Device if Device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL.to(self.DEVICE)
        self.OPTIMIZER = Optim if Optim else torch.optim.AdamW(MODEL.parameters(), lr=1e-4, weight_decay=1e-5)



            