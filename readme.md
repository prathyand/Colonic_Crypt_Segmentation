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

- Deeplabv3+ (`xception41` backbone)
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
| Deeplabv3+ |   xception41   |       12       |     0.1809     |          0.8671          |     0.1416    |

based on the test DICE score 'UNET' gives the best results on test DICE score, but mask prediction of UNET gives some unsatisfactory results.

## Mask Prediction output comparision

Even though UNET appears to be the best model based on the test dice score, it faces issues with correctly identifying the (absence of) crypts in the region where target mask doesn't have '1' class.

Example 1

![predictions of trained models on a patch](/report_images/0_CL_HandE_1234_B004_bottomleft.tiff.png)

Example 2

![predictions of trained models on a patch](/report_images/0_HandE_B005_CL_b_RGB_bottomleft.tiff.png)


In both the examples above, UNET++ and Deeplabv3+ perform much better in the regions of dominant background class because of UNET++ architecture improvements over traditional UNET (such as deep supervision - loss function accessible to shalow layers which results in better gradient flow). Deeplabv3+

Example 3

![predictions of trained models on a patch](/report_images/24_CL_HandE_1234_B004_bottomleft.tiff.png)

As soon as we see the results on a patch with dominant class '1', it becomes clear that UNET does a much better job of segmenting the crypts. Unet++ does a decent job of seperating two crypt mask boundries in close proximity, with DeepLabv3+ performing worse than the other two. 

### Prediction on test images

Gievn the performances of the models, inference on test images is done using only Unet++ and DeepLabv3+ due to UNET's poor performance in crypt deficient regions of the image. 

To perform the inference, test images are padded with `right_borders` and `bottom_borders` using opencv to match the dimentions so that **Non-Overlapping** patches of size 512,512 can be created. Average probability is calculated on each patch from using `UNET++` and `DeepLabv3+` output with sigmoid activation. Probabilities are then averaged over models and class for each pixel is predicted using 0.5 as threshold.

All the patch masks are then stitched back to recreate the full prediction mask. Below is the result on test images:

![predictions of trained models on a patch](/report_images/maskoverlays.png)

#### Test dataset DICE score results:

|         **test image**         | **Mean Dice** |
|:------------------------------:|:-------------:|
|  CL_HandE_1234_B004_bottomleft |     0.9174    |
| HandE_B005_CL_b_RGB_bottomleft |     0.8065    |


## t-SNE analysis on test and train datasets:

To find the feature overlap within train and test datasets, t-SNE analysis is used. Featues are extracted from the last layer of encoder in UNET++ for train and test datasets. Any pixel within the (512,512) contains a mask pixel, it t is considered as class '1'. Features are created for both test and train datasets and t-SNE is performed to project the data into two dimentions. 

Train data has class '0' datapoints to some extent clustered and seperated from the class '1' datapoints but there is a significant overlap in both classes in test dataset. This is probably due to the the labeling strategy described above. 

#### train data TSNE

![predictions of trained models on a patch](/report_images/train_tsne_.png)

#### test data TSNE

![predictions of trained models on a patch](/report_images/test_tsne_.png)

## CSV submission

`submission.csv` file contains the RLE encodings created for the test image prediction masks. 



