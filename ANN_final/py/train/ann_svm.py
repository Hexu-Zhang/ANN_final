# 启动：python3 ann_svm.py --train

import os
import torch
import numpy as np
import pandas as pd
import joblib
import pickle
import argparse
import scipy.io  # Read Matlab files
import matplotlib.pyplot as plt
import models.fourier_features as ff
from collections import Counter
from sklearn.decomposition import PCA
from sklearn import svm  # SVM
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error, classification_report, accuracy_score, confusion_matrix
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from neuro_tf_utils import *
from torch.utils.data import TensorDataset, DataLoader
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline# 初始化阶段：数据设定、设备加载、主函数选择

print("初始化阶段开始")
GHz = 1e9

# 获取目前使用的设备
def get_device():
    device_str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    return device_str

device = get_device()
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}")