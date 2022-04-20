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




class model_trainer:

    def __init__(self,conf,trainloader,testloader,EPOCHS=1):
        
        self.conf=conf
        self.trainloader=trainloader
        self.testloader=testloader
        self.EPOCHS=EPOCHS

    def testloss(self):
        self.conf.MODEL.eval()
        diceloss=0
        metric_sum=0.0
        metric_count=0
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        with torch.no_grad():
            for batch_data in self.testloader:
                input,mask=batch_data[0].to(self.conf.DEVICE),batch_data[1].to(self.conf.DEVICE)
                output=self.conf.MODEL(input)
                loss=self.conf.LOSS_F(output,mask)
                diceloss+=loss
                value, not_nans = dice_metric(y_pred=output, y=mask)
                not_nans = not_nans.mean().item()
                metric_count += not_nans
                metric_sum += value.mean().item() * not_nans
        
        return diceloss/(len(self.testloader)),metric_sum/metric_count



    def fit(self):
    
        epoch_loss_values = []
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

                mus=mask.unsqueeze(1)
                print(outputs.size(),mus.size())
                loss = self.conf.LOSS_F(outputs, mus)
                loss.backward()
                self.conf.OPTIMIZER.step()
                epoch_loss += loss.item()

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            if (epoch + 1) % 1 == 0:
                testloss,testDice=self.testloss()
                print(f"current epoch: {epoch + 1} Avg test Dice: {testDice:.4f} Avg test loss: {testloss:.4f}")

    