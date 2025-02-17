# 启动 python3 /home/zhanghexu/PJT/DR_Application/ANN_final/py/classfier/svm.py

import os
import numpy as np
import scipy.io
import joblib
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from collections import Counter
from imblearn.over_sampling import SMOTE


# 获取路径
cur_dir = os.path.dirname(os.path.realpath(__file__))

# 绝对路径
# 仅增加
#training_data_path = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/enhenced/Enhenced_Training_Data.mat"
#test_data_path = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/enhenced/Enhenced_Test_Data.mat"
# 删除少数项
training_data_path = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/deleted/Enhenced_Training_Data.mat"
test_data_path = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/deleted/Enhenced_Test_Data.mat"

# 仅增加
#VF_SAMPLES_PATH = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/processed_data/vf_samples.npy"
#VF_SAMPLES_TEST_PATH = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/processed_data/vf_samples_test.npy"

# 删减
VF_SAMPLES_PATH = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/deleted/vf_samples.npy"
VF_SAMPLES_TEST_PATH = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/deleted/vf_samples_test.npy"


SAVE_DIR = "/home/zhanghexu/PJT/DR_Application/ANN_final/checkpoint/svm_del"

# 检查文件存在
if not os.path.exists(training_data_path):
    raise FileNotFoundError(f"Training data file not found: {training_data_path}")
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test data file not found: {test_data_path}")

# 加载文件
training_data = scipy.io.loadmat(training_data_path)
test_data = scipy.io.loadmat(test_data_path)

# 提取数据
X = training_data["training_samples"]
Y = training_data["training_labels"]
X_test = test_data["test_samples"]
Y_test = test_data["test_labels"]

# 归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 加载数据
vf_samples = np.load(VF_SAMPLES_PATH, allow_pickle=True)  # 加载训练数据
vf_samples_test = np.load(VF_SAMPLES_TEST_PATH, allow_pickle=True)  # 加载测试数据

model_orders_observed = [len(vf.poles) for vf in vf_samples]
model_orders_test_observed = [len(vf.poles) for vf in vf_samples_test]

# 1. 保存数据到指定路径
save_path = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/enhenced"

np.save(f"{save_path}/model_orders_observed.npy", model_orders_observed)
np.save(f"{save_path}/model_orders_test_observed.npy", model_orders_test_observed)

# 2. 统计各个阶数在训练集及测试集中出现的次数
# 统计训练集
count_observed = Counter(model_orders_observed)
print("Original data distribution (model_orders_observed):")
print(f"Counter({count_observed})")

# 统计测试集
count_test_observed = Counter(model_orders_test_observed)
print("Original data distribution (model_orders_test_observed):")
print(f"Counter({count_test_observed})")

print("SVM训练开始")

# 定义 SVM 模型（使用 GridSearchCV 进行超参数调优）
class_weights = {
    6: 1,
    7: 3,
    8: 5,
    9: 3,
    10: 3,
    11: 1
}
#初始化 SVM 模型，使用最优参数
svc = svm.SVC(
    C=5,  # 最优参数
    gamma=4.9,  # 最优参数
    kernel='rbf',  # 最优参数
    degree=2,  # 最优参数
    class_weight=class_weights,
    probability=True,
    random_state=1
)

#svc = svm.SVC(
#    C=0.56,  # 最优参数
#    gamma= 1.88,  # 最优参数
#    kernel='rbf',  # 最优参数
#    degree=1,  # 最优参数
#    class_weight=class_weights,
#    probability=True,
#    random_state=1
#)

svc = svm.SVC(class_weight=class_weights, probability=True, random_state=1)

## 超参数调优
#param_grid = {
    #'C': [0.01,0.1,1,5,10,15],
    #'C': [0.01,0.1,1,2,3,4],
    #'C': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
    #'C': [0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69],
    #'C': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,01.4,0.15,0.16,0.17,0.18,0.19, 1],
    #'C': [0.1, 1,2,3,4,5,6,7,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9, 10,11,12,13,14,15,16,17,18,19,20, 100, 1000],
    #'C': [ 7.5, 7.51,7.52,7.53,7.54,7.55,7.56,7.57,7.58,7.59,7.6,7.61,7.62,7.63,7.64,7.65,7.66,7.67,7.68,7.69, 7.7,7.71,7.72,7.73,7.74,7.75,7.76,7.77,7.78,7.79,7.80],
    #'C': [ 1.2,1.21,1.22,1.23,1.24,1.25,1.26,1.27,1.28,1.29,1.3,1.31,1.32,1.33,1.34,1.35,1.36,1.37,1.38,1.39,1.4],
    #'C': [ 1,2,3,4,5,6,7,7.5, 7.51,7.52,7.53,7.54,7.55,7.56,7.57,7.58,7.59,7.6,7.61,7.62,7.63,7.64,7.65,7.66,7.67,7.68,7.69, 7.7],
    #'C': [ 1,2,3,3.1,3.2,3.3,3.4,3.41,3.42,3.43,3.44,3.45,3.46,3.47,3.48,3.49,3.5,3.51,3.52,3.53,3.54,3.55,3.56,3.57,3.58,3.59,3.6,3.7,3.8,3.9,4,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5,6,7],
    #'C': [ 5,6,7,8,9,9.0,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,10,10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11,12,13,14,15,16,17,18,19,20 ],
    #'gamma': ['scale', 'auto',0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0],
    #'gamma': ['scale', 'auto', 1.71,1.72,1.73,1.74,1.75,1.76,1.77,1.78,1.79,1.80,1.81,1.82,1.83,1.84,1.85,1.86,1.87,1.88,1.89,1.90],
    #'gamma': ['scale', 'auto', 4,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6],
    #'gamma': ['scale', 'auto', 4.8,4.81,4.82,4.83,4.84,4.85,4.86,4.87,4.88,4.89,4.9,4.91,4.92,4.93,4.94,4.95,4.96,4.97,4.98,4.99,],
    #'gamma': ['scale', 'auto', 9,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,9,10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11],
   # 'kernel': ['rbf', 'linear'],
    #'kernel': ['rbf', 'linear', 'poly'],
    #'degree': [1, 2, 3, 4],  # 添加 degree 参数
#}

#grid_search = GridSearchCV(svc, param_grid, cv=StratifiedKFold(5), scoring='f1_macro', n_jobs=-1, verbose=2)
#grid_search.fit(X_scaled, model_orders_observed)
#best_svm = grid_search.best_estimator_

#print(f"Best Parameters: {grid_search.best_params_}")

# 使用交叉验证评估模型
#print("正在进行交叉验证...")
#scores = grid_search.cv_results_['mean_test_score']
#print(f"Cross-validation accuracy (best): {max(scores)*100:.2f}%")

# 处理数据不平衡（过采样）
smote = SMOTE()
X_smote, Y_smote = smote.fit_resample(X_scaled, model_orders_observed)

# 使用超参数训练模型
#best_svm.fit(X_smote, Y_smote)
svc.fit(X_smote, Y_smote)

print("使用最优参数的 SVM 模型训练完成！")
print(f"使用的参数: C=5, gamma=4.9, kernel='rbf', degree=2")

# 使用进度条显示测试结果
print("对测试集进行预测...")
test_predictions = []
for sample in tqdm(X_test_scaled):
    #test_predictions.append(best_svm.predict([sample])[0])
    test_predictions.append(svc.predict([sample])[0])

test_accuracy = accuracy_score(model_orders_test_observed, test_predictions)
print(f"Testing SVM Accuracy: {test_accuracy * 100:.2f}%")

# 确保目录存在
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 保存模型
SAVE_PATH = os.path.join(SAVE_DIR, "svm.pkl")
joblib.dump(svc, SAVE_PATH)
print(f"模型已保存到: {SAVE_PATH}")

# 保存预处理器
SCALER_SAVE_PATH = os.path.join(SAVE_DIR, "scaler.pkl")
joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"预处理器已保存到: {SCALER_SAVE_PATH}")

# 输出评估结果
print(f"Test Actual Model Orders: {model_orders_test_observed}")
print(f"Test Predicted Model Orders: {test_predictions}")

# 评估测试数据
test_conf_matrix = confusion_matrix(model_orders_test_observed, test_predictions)
test_cls_report = classification_report(model_orders_test_observed, test_predictions, zero_division=0)
print(f"Testing SVM Confusion Matrix\n", test_conf_matrix)
print(f"Testing SVM Classification Report\n", test_cls_report)