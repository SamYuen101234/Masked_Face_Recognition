# Masked Facial Recognition
2020/2021 HKUST CSE FYP Masked Facial Recognition, developer: Sam Yuen, Alex Xie, Tony Cheng

Supervisor: Prof. Dit-Yan YEUNG

This Github repository shows how to train a masked face recognition model. We use the masked facial recognition models to build a building security system in another [Github repository](https://github.com/SamYuen101234/BSMFR).

### Download masked facial recognition train and test data


| Dataset     | # Identities|# Images |
| ----------- | ----------- | ----------- |
| CASIA      | 10,585       | 492,832       |
| VGG2   | 8,631        | 2,024,897        |
| LFW (eval+test)   | 5,749        | 64,811        |

- Information of all datasets shown above is after preprocessing so the numbers and sizes differ from the original datasets

1. Training set 1 (Masked CASIA) 3GB
Link: [Google Drive](https://drive.google.com/file/d/1wqLxMbV335dHkX_7lLCROABd6-aC149B/view?usp=sharing)

2. Training set 2 (Masked VGG2) 11GB
Link: [Google Drive](https://drive.google.com/file/d/1qPwssuYgieeM_38gBJ9N-s-S1ySRo9Ka/view?usp=sharing)

3. Training CSV (Masked CASIA + Masked VGG2) 115MB
Link: [Google Drive](https://drive.google.com/file/d/1BrZtoDbdc61pdH2GQ4v1GJKr9cQM-jVh/view?usp=sharing)

3. Test set (LFW), including evaluation set and test set, 148MB
Link: [Google Drive](https://drive.google.com/file/d/1bVP3gbAHjzB33wkEbG8MChd7tp1Ee2Oo/view?usp=sharing)

4. Evaluation set 1 CSV  (all same person pairs) 7MB
Link: [Google Drive](https://drive.google.com/file/d/1f3NTKV4YxOmHsZHgEm9MTacW85fyiZRM/view?usp=sharing)

5. Evaluation set 2 CSV (all different person pairs) 7MB
Link: [Google Drive](https://drive.google.com/file/d/1RdFbIGDiMMVaAPpt8CsykJRqTE4GfJH3/view?usp=sharing)

7. Test CSV, 782KB
Link: [Google Drive](https://drive.google.com/file/d/15axXyvMhlu4z3_jAZJh16f5wucoOS_Jd/view?usp=sharing)

### Download Pre-train models

| Models     | # Architect|# Loss func |# Pre-trained |# Acc |
| ----------- | ----------- | ----------- |----------- |----------- |
| Model1      | InceptionResNetV1       | ArcFace with focal loss     |Yes      |95.xx%       |
| Model2      | InceptionResNetV1        | Triplet loss with online triplet mining        |Yes             |94.xx%       |
| Model3      | SE-ResNeXt-101        | ArcFace with focal loss     |No      |93.xx%       |


Model1
Link: [Google Drive]()

Model2:
Link: [Google Drive]()

Model3:
Link: [Google Drive]()

If you want to know more about the training process and concept, you can read our progress report and the following papers:

1. [Our progress report (Implementation section)](https://drive.google.com/file/d/17qEgb0ZC0Ml7gym4rl2ShGBbrjmXATQz/view?usp=sharing)
2. [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
3. [Offline versus Online Triplet Mining based on Extreme Distances of Histopathology Patches](https://arxiv.org/abs/2007.02200)
4. [Masked Face Recognition for Secure Authentication](https://arxiv.org/abs/2008.11104)
5. [Deep metric learning using Triplet network](https://arxiv.org/abs/1412.6622)
6. [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
7. [Normal Face Recignition with ArcFace in Pytorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)

You can find more in the reference list.

### Run

Before you run you need to install the follow package or library first:
> pip install tqdm
> pip install facenet-pytorch
> pip install pip install efficientnet
> pip install timm

We expect that you have install other common packages like torch, numpy, pandas...

To train a model with online triplet mining, run:
> python3 main.py

To train a model with ArcFace, run:
> python3 main2.py

### Methodolegy


To know more our training methods in details, please read our progress report and the paper in reference.

### Result

### Reference list

Only show some important reference:

### License

