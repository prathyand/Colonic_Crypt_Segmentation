import tifffile
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import os
import math

TRAIN_PATH_ORIG ="Colonic_crypt_dataset/train/"
TEST_PATH_ORIG = "Colonic_crypt_dataset/test/"
TRAINMASK_ORIG = "Colonic_crypt_dataset/train_mask/"
TESTMASK_ORIG = "Colonic_crypt_dataset/test_mask/"
TRAIN_IMGPATCH_SAVEPATH = "data_process/train_img_patch/"
TRAIN_MASKPATCH_SAVEPATH = "data_process/train_mask_patch/"
TRAIN_IMAGES_PATHLIST = [TRAIN_PATH_ORIG+i for i in os.listdir(TRAIN_PATH_ORIG) if ".tiff" in i]
TRAIN_MASK_PATHLIST = [TRAINMASK_ORIG+i for i in os.listdir(TRAINMASK_ORIG) if ".tiff" in i]

def getImg(imgpath:str,plotImg:bool=False)->np.ndarray:
    img = tifffile.imread(imgpath)
    if plotImg:
        print(img.shape)
        plt.imshow(img)
        plt.show()
    return img

def isUsefulPatch(img:np.ndarray)->bool:
    ## Image shape is (H,W,3)
    H,W=img.shape[0],img.shape[1]
    whiteregions=img>=220
    whiteregions=np.sum(whiteregions[:,:,0]*whiteregions[:,:,1]*whiteregions[:,:,2])/(H*W)
    blackregions=img<=60
    blackregions=np.sum(blackregions[:,:,0]*blackregions[:,:,1]*blackregions[:,:,2])/(H*W)

    if blackregions>=0.3 or whiteregions>=0.65:
        return False
    
    return True

def Patchhasannotation(img:np.ndarray)->bool:
    ## Image shape is (H,W,3)
    if np.max(np.array(img))==0:
        return False
    return True


def extract_patches(x,PatchShape,Stride):
        im_h = x.shape[0] 
        im_w = x.shape[1]

        def get_patch(x, ptx,PatchShape,Stride):
            pty = (ptx[0]+PatchShape[0],
                ptx[1]+PatchShape[1])
            win = x[ptx[0]:pty[0], 
                    ptx[1]:pty[1]]
            return win

        def extract_infos(length, win_size, step_size):
            flag = (length - win_size) % step_size != 0
            last_step = math.floor((length - win_size) / step_size)
            last_step = (last_step + 1) * step_size
            return flag, last_step

        h_flag, h_last = extract_infos(im_h, PatchShape[0], Stride[0])
        w_flag, w_last = extract_infos(im_w, PatchShape[1], Stride[1])    

        sub_patches = []
        for row in range(0, h_last, Stride[0]):
            for col in range(0, w_last, Stride[1]):
                win = get_patch(x, (row, col),PatchShape,Stride)
                sub_patches.append(win)  

        return sub_patches


def generatePatches(inputImagedirPath:str,outputImagedirPath:str,inputmaskdirPath:str,outputmaskdirPath:str,PatchShape:tuple,PatchStride:tuple): 
    Imglst = [i for i in os.listdir(inputImagedirPath) if ".tiff" in i]

    for i in Imglst:
        print(inputImagedirPath+i)
        image=getImg(inputImagedirPath+i)
        mask=getImg(inputmaskdirPath+i)
        imgpatchlst=extract_patches(image,PatchShape,PatchStride)
        maskpatchlst=extract_patches(np.expand_dims(mask,axis=2),PatchShape,PatchStride)

        for j in range(len(imgpatchlst)):
            if(Patchhasannotation(maskpatchlst[j]) and isUsefulPatch(imgpatchlst[j])):
                io.imsave(outputImagedirPath+str(j)+i,imgpatchlst[j],check_contrast=False)
                io.imsave(outputmaskdirPath+str(j)+i,maskpatchlst[j],check_contrast=False)

    return