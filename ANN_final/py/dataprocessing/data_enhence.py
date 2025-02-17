# 启动 python3 /home/zhanghexu/PJT/DR_Application/ANN_final/py/dataprocessing/data_enhence.py

import os
import numpy as np
import scipy.io
import json
import joblib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from collections import defaultdict

print("数据强化")

# 获取路径
cur_dir = os.path.dirname(os.path.realpath(__file__))

# 加载对应路径的文件
training_data_path = os.path.join(cur_dir, "/home/zhanghexu/PJT/DR_Application/ANN_final/data/original/Training_Data.mat")
test_data_path = os.path.join(cur_dir, "/home/zhanghexu/PJT/DR_Application/ANN_final/data/original/Real_Test_Data.mat")

# 定义数据存储路径
DATA_DIR = os.path.join(cur_dir, 'data')
PROCESSED_DATA_DIR = '/home/zhanghexu/PJT/DR_Application/ANN_final/data/processed_data'  # 修改为指定路径

# 检查文件存在
if not os.path.exists(training_data_path):
    raise FileNotFoundError(f"Training data file not found: {training_data_path}")
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test data file not found: {test_data_path}")

# 加载文件
training_data = scipy.io.loadmat(training_data_path)
test_data = scipy.io.loadmat(test_data_path)

# 检查文件内容
required_keys = ["candidates", "responses"]
for key in required_keys:
    if key not in training_data:
        raise ValueError(f"Training data is missing key: {key}")
required_keys = ["real_test_candidates", "real_test_responses"]
for key in required_keys:
    if key not in test_data:
        raise ValueError(f"Training data is missing key: {key}")
print("文件提取完成")

# 提取数据
# X : [lp @ ln @ hc]^T
# Y : 频率、实部、虚部
print("数据加载阶段开始")
X = training_data["candidates"]
Y = training_data["responses"]
X_test = test_data["real_test_candidates"]
Y_test = test_data["real_test_responses"]

# 加载数据
VF_SAMPLES_PATH = '/home/zhanghexu/PJT/DR_Application/ANN_final/data/processed_data/vf_samples.npy'
VF_SAMPLES_TEST_PATH = '/home/zhanghexu/PJT/DR_Application/ANN_final/data/processed_data/vf_samples_test.npy'

vf_samples = np.load(VF_SAMPLES_PATH, allow_pickle=True)  # 加载训练数据
vf_samples_test = np.load(VF_SAMPLES_TEST_PATH, allow_pickle=True)  # 加载测试数据

model_orders_observed = [len(vf.poles) for vf in vf_samples]
model_orders_test_observed = [len(vf.poles) for vf in vf_samples_test]

def save_model_orders(model_orders, filename):
    with open(os.path.join(PROCESSED_DATA_DIR, filename), 'w') as f:
        json.dump(model_orders, f)

save_model_orders(model_orders_observed, 'model_orders_observed.json')
save_model_orders(model_orders_test_observed, 'model_orders_test_observed.json')

def count_orders(model_orders):
    order_counts = defaultdict(int)
    for order in model_orders:
        order_counts[order] += 1
    return dict(order_counts)

order_counts_observed = count_orders(model_orders_observed)
order_counts_test = count_orders(model_orders_test_observed)

# 保存统计结果
with open(os.path.join(PROCESSED_DATA_DIR, 'order_counts_observed.json'), 'w') as f:
    json.dump(order_counts_observed, f)

with open(os.path.join(PROCESSED_DATA_DIR, 'order_counts_test.json'), 'w') as f:
    json.dump(order_counts_test, f)


