

# Diabetic Retinopathy Detection
This repository contains the code for training and evaluation of Deep Learning Lab work.

## Train and Evaluation

### Datasets

Pre-process the [*IDRID*]([https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)) datasets using the code-base at `Input_pipeline/` (https://github.com/TWWinde/Diabetic_Retinopathy/tree/main/diabetic_retinopathy/Input_pipeline).
# How to run the code

- Change  `batch.sh` to `python3 main.py` Run `main.py` to train or evaluation. Run `main.py` ,it will process the image and serialize images and labels into the TFRecord format and then training the model automatically.      
- Change **FLAGS** in main.py to decide whether to train or test.  change **model_flag** to decide to train or evaluate which model. change **test_flag** to choose the method of evaluation, eg. evaluatatin, confusionmatrix, Dimensionality_Reduction, ROC.
  ``` python
  Choose_model = ['vgg_like', 'resnet', 'tl_inception', 'tl_xception', 'tl_inception_resnet']
  model_flag = Choose_model[0]
  Choose = ['evaluate_fl', 'confusionmatrix', 'Dimensionality_Reduction', 'ROC']
  test_flag = Choose[0]
  ```
- Change  `batch.sh` to `python3 tune.py`. Run `tune.py` to do hyperparameter optimization. 



## Results of Diabetic retinopathy
- The results of image pre-processing (before and after). 
```python
img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)
```
<img src="https://media.github.tik.uni-stuttgart.de/user/5018/files/89b9d1a1-ad95-42cb-b43d-7200d0daaf60" width="350px"> <img src="https://media.github.tik.uni-stuttgart.de/user/5018/files/d089052f-fa81-47c2-bd6d-a52404eb432e" width="230px">

### Training results
| Model      |  Test Accuracy  | Total Params | AUC   |
|:----------:|:-----------------: |:------------:|:------------:|
| Basic-CNN  |      0.7282     |    397,986   | 0.82  |
|  ResNet-18  |   0.7864       |   11,559,970    | 0.86   |
|    VGGNet  |    0.8447      |    4,736,402   | 0.92   |
| XceptionV3  |      0.7961    |  22,335,618  | - |
|  InceptionV3 | 0.8155      |    21,394,314   | - |
| Inception-resnet|    0.8058   | 54,738,498  | - |

- The train and validation accuracy and loss for VGG  
<img src="https://media.github.tik.uni-stuttgart.de/user/5018/files/d31dade3-0d37-4882-a1ef-abd4b2c9a843" width="360px"><img src="https://media.github.tik.uni-stuttgart.de/user/5018/files/2686bba5-13a7-47cc-8b75-63f5049b5567" width="380px">
      
 - The reults of Hyperparameter optimization on VGGNet     
<img width="500px" alt="vgg hyper" src="https://media.github.tik.uni-stuttgart.de/user/5018/files/d9390a68-24be-4372-92d7-d9056f0ba42e">    

- The comparation of different parameters of focal loss, and the hyperparameter optimization
<img src="https://media.github.tik.uni-stuttgart.de/user/5018/files/0c7f91da-943f-4690-bcb5-702f400985df" width="400px">  <img src="https://media.github.tik.uni-stuttgart.de/user/5018/files/046d91b6-37ff-4f53-80eb-f867366bda2c" width="380px">   
 
- The result of VGGNet with 5 blocks with 84.46% accuracy, and the results of ensemble learning with 86.46% accuracy 
                                                 
<img src="https://media.github.tik.uni-stuttgart.de/user/5018/files/1261abaf-cecd-4eef-984d-7a6ba40e806e" width="350px"><img src="https://media.github.tik.uni-stuttgart.de/user/5018/files/36ae92db-6bb8-410e-914d-808b4c185504" width="380px">      
  

### Visualization:Grad-CAM and dimentionality reduction.   
 
<img src="https://media.github.tik.uni-stuttgart.de/user/5018/files/e8ae5183-35d9-43b8-b839-1bb63496e1d1" width="300px">                  <img src="https://media.github.tik.uni-stuttgart.de/user/5018/files/e6074914-ec04-4ce3-9e48-32f484e50415" width="330px">


