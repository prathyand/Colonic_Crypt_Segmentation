import numpy as np
import pandas as pd
import predict
import datapreprocessing as dapre
import tifffile
import dataset
from albumentations.pytorch.transforms import ToTensorV2
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose
)
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import os
import torch
import trainer
import model_config
from torch.utils.data import DataLoader,Dataset
import monai
from monai.losses import DiceLoss, DiceCELoss
import segmentation_models_pytorch as smp
from empatches import EMPatches
import skimage.io as io
from monai.visualize.utils import blend_images

def load_models(modeldir,modellist):
    models={}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for mdname in modellist:
        if mdname=="UNET":
            model= smp.Unet(encoder_name='efficientnet-b2',in_channels=3,classes=1,
            encoder_weights='imagenet')
            model.load_state_dict(torch.load(modeldir+"Unet.pth"))
            model.to(device=device)
            models["UNET"]=model

        if mdname=="DeepLab":
            model=smp.DeepLabV3Plus(encoder_name='tu-xception41',in_channels=3,classes=1,encoder_output_stride=16,
            encoder_weights='imagenet')
            model.load_state_dict(torch.load(modeldir+"DeepLabV3Plus.pth"))
            model.to(device=device)
            models["DeepLab"]=model

        if mdname=="UNET++":
            model=smp.UnetPlusPlus(encoder_name='efficientnet-b2',in_channels=3,classes=1,
            encoder_weights='imagenet')
            model.load_state_dict(torch.load(modeldir+"UnetPlusPlus.pth"))
            model.to(device=device)
            models["UNET++"]=model
            
    return models


def predictPatchlist(testdir,models):
  
    testimgpreds=[]
    testimglist=os.listdir(testdir)
    emp = EMPatches()
    post_trans = Compose(
                [Activations(sigmoid =True)]
            )
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ind,ti in enumerate(testimglist):
        if ".tiff" not in ti:
          continue
        print(ti)
        ig=tifffile.imread(testdir+ti)
        igmk=tifffile.imread(dapre.TESTMASK_ORIG+ti)

        ig2=cv2.copyMakeBorder(
                 ig, 
                 0, 
                 72, 
                 0, 
                 416, 
                 cv2.BORDER_CONSTANT, 
                 value=0
              )
        ig2mk=cv2.copyMakeBorder(
            igmk, 
            0, 
            72, 
            0, 
            416, 
            cv2.BORDER_CONSTANT, 
            value=0
        )
        patches, indices = emp.extract_patches(ig2, patchsize=512, overlap=0)
        patchesmk, indices = emp.extract_patches(ig2mk, patchsize=512, overlap=0)
        patchpred=[]
        
        for i,img in enumerate(patches):
            # print("i",i)
            # save images
            if i in [0,24,70]:
                fig=plt.figure(figsize=(16, 16))
                rows = 2
                cols = 2
                axes=[]
                a=0
              
            outavg=None

            img=dataset.TEST_TRANSFORMS(image=img)['image']
            if i in [0,24,70]:
                maskA=dataset.TEST_TRANSFORMS(image=patchesmk[i])['image']
                bi=blend_images(img,label=maskA,alpha=0.5)
                bi=bi.permute(1,2,0)
                axes.append(fig.add_subplot(rows, cols, a+1))
                subplot_title=("Patch:"+ti)
                axes[-1].set_title(subplot_title) 
                plt.imshow(bi)
                a+=1
            


            img=img.float()
            img = img.unsqueeze(0)
            img=img.to(device=device)
            for mdname,model in models.items():
                model.eval()
                with torch.no_grad():
                # print("imageshape: ",img.shape,"device:",device)
                    outmsk=model(img)
                    # plt.subplot(1,5,k)
                    # plt.title(mdname)
                    if i in [0,24,70]:
                        mp=torch.sigmoid(outmsk.squeeze()).detach().cpu()
                        axes.append(fig.add_subplot(rows, cols, a+1))
                        subplot_title=(mdname+": "+ti)
                        axes[-1].set_title(subplot_title) 
                        plt.imshow(mp)
                        a+=1
                    # plt.show()
                    outmsk=post_trans(outmsk)
                    if outavg is None:
                        outavg=outmsk
                    else:
                        outavg+=outmsk
            if i in [0,24,70]:
                fig.tight_layout() 
                plt.savefig('report_images/'+str(i)+'_'+ti+'.png')   
                # plt.show()
                    

            outavg=outavg/len(models)
            # print(outavg.shape,torch.min(outavg),torch.max(outavg))
            outavg=outavg.squeeze().detach().cpu().numpy()>0.5
            outavg=outavg*255.
            outavg=np.repeat(outavg[:, :, np.newaxis], 3, axis=2)
            patchpred.append(outavg)
            # print(outavg.shape,np.min(outavg),np.max(outavg))
        merged_img = emp.merge_patches(patchpred, indices)
        merged_img=merged_img[:(ig.shape[0]),:(ig.shape[1]),:]
        testimgpreds.append((merged_img,ti))
        io.imsave("predictedtestMask/"+ti,merged_img,check_contrast=False)
    
    return testimgpreds





