import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import monai
from monai.networks.nets import unetr
from monai.networks.nets import UNet
import model_config as mc
from monai.metrics import DiceMetric
import os
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose
)
 



class model_trainer:

    def __init__(self,conf,trainloader,testloader,EPOCHS=20,modelpath="modelsaved/"):
        
        self.conf=conf
        self.trainloader=trainloader
        self.testloader=testloader
        self.EPOCHS=EPOCHS
        self.modelpath=modelpath
        
        self.trainloader.smp_preproc = self.conf.preproc
        self.testloader.smp_preproc = self.conf.preproc

    def testloss(self):
        self.conf.MODEL.eval()
        diceloss=0
        metric_sum=0.0
        metric_count=0
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        post_trans = Compose(
                [Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
            )
        with torch.no_grad():
            for batch_data in self.testloader:
                input,mask=batch_data[0].to(self.conf.DEVICE),batch_data[1].to(self.conf.DEVICE)
                output=self.conf.MODEL(input)
                loss=self.conf.LOSS_F(output,mask.unsqueeze(1))
                output = post_trans(output)
                diceloss+=loss
                value = dice_metric(y_pred=output, y=mask.unsqueeze(1))
                metric_count += len(value)
                metric_sum += value.sum().item()
        
        return diceloss/(len(self.testloader)),metric_sum/metric_count

    def fit(self):
    
        epoch_loss_values = []
        best_metric=0
        best_metric_epoch=0
        for epoch in range(self.EPOCHS):
            print(f"epoch {epoch + 1}/{self.EPOCHS}")
            self.conf.MODEL.train()
            epoch_loss = 0
            step = 0
            for batch_data in self.trainloader:
                step += 1
                input, mask = (
                    batch_data[0].to(self.conf.DEVICE),
                    batch_data[1].to(self.conf.DEVICE),
                )
                self.conf.OPTIMIZER.zero_grad()
                outputs = self.conf.MODEL(input)
                loss = self.conf.LOSS_F(outputs, mask.unsqueeze(1))
                loss.backward()
                self.conf.OPTIMIZER.step()
                epoch_loss += loss.item()

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            if (epoch + 1) % 1 == 0:
                testloss,testDice=self.testloss()
                print(f"current epoch: {epoch + 1} Avg test Dice: {testDice:.4f} Avg test loss: {testloss:.4f}")

                if testDice > best_metric:

                    best_metric = testDice
                    best_metric_epoch = epoch + 1
                    torch.save(self.conf.MODEL.state_dict(),os.path.join(self.modelpath, self.conf.MODEL.__class__.__name__+".pth"))
                    print("saved new best metric model")

    