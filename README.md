# Breast-Cancer-Classification
COS429 Final Project

Model Codes are written in mini_ddsm_\*.py files:
- mini_ddsm_cnn.py -> SimpleCNN
- mini_ddsm_vgg_2.py -> VGG-16
- mini_ddsm_resnet.py -> ResNet-50
- mini_ddsm_vgg2_oversample.py -> Oversampling Method 1
- mini_ddsm_vgg_data2.py -> Data Augmentation Method 2
- mini_ddsm_vgg2_augment.py -> Data Augmentation Method 3
- mini_ddsm_vgg_uniform.py -> Data Augmentation Method 4

Results are in corresponding df_\*.csv files

Mini_DDSM.ipynb contains data analysis, plots, and calculations

Failed Ideas:
- Contrast, brightness, and ROI cropping in Mini_DDSM.ipynb 
- Running a model such that we group images by patient is in mini_ddsm_vgg_bypatient.py
- Running a model augmented with brighter images is in mini_ddsm_vgg_bright.py
- Running a model augmented with contrast added to images is in mini_ddsm_vgg_contrast.py
- Other fairness metrics such as demographic parity, equalized odds, disparate impact, etc in Mini_DDSM.ipynb
- Simple attempt at using a DCGAN to generate new images as a form of data augmentation in dcgan.py
