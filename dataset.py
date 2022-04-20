import torch
from torch.utils.data import DataLoader,Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import os
import numpy as np
import pandas as pd
import tifffile

TRAIN_TRANSFORMS = A.Compose([A.OneOf([
            A.ShiftScaleRotate(p=0.3),
            A.ElasticTransform(p=0.3),
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.3),
            A.NoOp()
        ]),
        A.ISONoise(p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.NoOp()
        ]),
        A.OneOf([
            A.FancyPCA(p=0.3),
            A.RGBShift(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.ToGray(p=0.2),
            A.NoOp()
        ]),
        A.OneOf([
                A.GaussianBlur(p=.3),
                A.GaussNoise(p=.3),
                A.PiecewiseAffine(p=0.3),
            ], p=0.3),
        A.ChannelDropout(p=0.3),
        A.RandomGridShuffle(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Transpose(p=0.3),
        ToTensorV2()
        ])

TEST_TRANSFORMS = A.Compose([ToTensorV2()])


class TissueData(Dataset):

    def __init__(self,imgpath,maskpath,transform,smp_preproc=None):
        self.imgpath=imgpath
        self.maskpath=maskpath
        self.transform=transform
        self.smp_preproc=smp_preproc

        self.imglist = os.listdir(imgpath)
        

    def __getitem__(self,idx):

        img = tifffile.imread(self.imgpath+self.imglist[idx])
        msk = tifffile.imread(self.maskpath+self.imglist[idx])

        if self.transform:
             transformed= self.transform(image=img,mask=msk)
             

        if self.smp_preproc:
            transformed = self.smp_preproc(image=transformed['image'],mask=transformed['mask'])
        
        img,msk =transformed['image'],transformed['mask']

        return img,(msk[:,:,0]/255.).type(torch.LongTensor)

    def __len__(self):
        return len(self.imglist)





