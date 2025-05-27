# 312551185_HW4
Student ID: 312551185\
Name: 郭晏涵

## Introduction
This project aims to tackle an image restoration problem focused on removing rain and snow degradations. The dataset consists of RGB images, with 1600 degraded-clean image pairs per weather type for training and validation, and 50 degraded images per type for testing. The objective is to restore each degraded image to its corresponding clean version.

## How to Install
```bash
conda create -n env python=3.12
conda activate env
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pycocotools==2.0.8 -c pytorch -c nvidia
```

Download PromptIR
```bash
git clone https://github.com/va1shn9v/PromptIR.git
```

## How to Run the Code
Train and Inference
```bash
python main.py
```

Create pred.npz
```bash
python img2npz.py
```

## Performance Snapshot
![image](https://github.com/slovengel/312551185_HW4/blob/main/codabench_snapshot.PNG)
