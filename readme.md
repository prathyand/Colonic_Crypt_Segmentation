# Colonic Crypts Segmentation 

### Train segmentation models (using Pytorch) to segment the colonic crypts in the tissue images

Dataset : https://drive.google.com/file/d/1RHQjRav-Kw1CWT30iLJMDQ2hPMyIwAUi/view?usp=sharing

Dependency requirements:
```!pip install segmentation_models_pytorch
!pip install monai
!pip install albumentations==1.1.0
!pip install tifffile
!pip uninstall opencv-python-headless
!pip install opencv-python==4.5.5.64
!pip list |findstr opencv
!pip install 'monai[all]'
!pip install empatches
!pip install segmentation-mask-overlay
!pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

```

## Data preperation

- Dataset(train) has 5 RGB images of size (4536, 4704)
- Train images are first split into overlapping patches of size (512,512) and stride of 256
- sample train image:
 
  ![Sample train image](/report_images/img.PNG)

- Patches are ignore if:
  - they contain black stripes (as seen above)
  - Their curresponding patches in the mask does not contain annotations (class '1') 

Different Image augmentation are used during training, implemented in `dataset.py` module, library used `albumentations`

## Model training
Libraries used for creating models: `segmentation_models_pytorch`,`monai` 

### Models

Four different models were trained , each for 20 epochs (except for `UNET++`):

- UNET (`efficientnetb2` backbone)
```
model = smp.Unet(encoder_name='efficientnet-b2',in_channels=3,classes=1,
            encoder_weights='imagenet')
```

- UNETR (Transformer based UNET : https://arxiv.org/abs/2103.10504)
```
model = monai.networks.nets.UNETR(spatial_dims=2,
            img_size=512,
            in_channels=3,
            out_channels=1,
            feature_size=16, 
            num_heads=6,
            dropout_rate=0.2,
            hidden_size=384,
            mlp_dim=768,
            norm_name='batch'
            )
```

- UNET++ (`efficientnetb2` backbone)
```
model = smp.UnetPlusPlus(encoder_name='efficientnet-b2',in_channels=3,classes=1,
            encoder_weights='imagenet')
```

- DeepLabV3++ (`xception41` backbone)
```
model = smp.DeepLabV3Plus(encoder_name='tu-xception41',in_channels=3,classes=1,encoder_output_stride=16,
            encoder_weights='imagenet')
```
 
### Loss function

Weighted sum of both Dice loss and Cross Entropy Loss `DiceCELoss` is used for parameter training.

AdamW optimizer is used for all models with following config:
```
AdamW(MODEL.parameters(), lr=1e-4, weight_decay=1e-5)
```

### Model performance metrics comparision
Model training is done in `train_models.ipynb`.

All models were trained for 20 epochs, except for `UNET++` which was trained for 5 more epochs with varying batch sizes (based on GPU memory, model parameters etc)

|  **Model**  |  **backbone**  | **Batch Size** | **train loss** | **best Test DICE score** | **test loss** |
|:-----------:|:--------------:|:--------------:|:--------------:|:------------------------:|:-------------:|
|     UNET    | efficientnetb2 |       12       |     0.2224     |          0.8732          |     0.1524    |
|    UNETR    |        -       |        6       |     0.6619     |          0.5130          |     0.5011    |
|    UNET++   | efficientnetb2 |       12       |     0.1880     |          0.8660          |     0.1422    |
| DeepLabv3++ |   xception41   |       12       |     0.1809     |          0.8671          |     0.1416    |

based on the test DICE score 'UNET' gives the best results on test DICE score, but mask prediction of UNET gives some unsatisfactory results.

## Mask Prediction output comparision

Even though UNET appears to be the best model based on the test dice score, it faces issues with correctly identifying the (absence of) crypts in the region where target mask doesn't have '1' class.

![predictions of trained models on a patch](/report_images/0_CL_HandE_1234_B004_bottomleft.tiff.png)


