# 启动 python3 /home/zhanghexu/PJT/DR_Application/ANN_final/py/dataprocessing/vector_fitting.py

import os
import sys
import skrf
import scipy.io
import torch
import numpy as np
import matplotlib.pyplot as mplt
# 加载 .mat 文件
from sklearn.preprocessing import StandardScaler  # 数据标准化工具
from builtins import ValueError, FileNotFoundError  # 内置异常类

# 获取路径
cur_dir = os.path.dirname(os.path.realpath(__file__))

# 加载对应路径的文件
training_data_path = os.path.join(cur_dir, "/home/zhanghexu/PJT/DR_Application/ANN_final/data/deleted/Enhenced_Training_Data.mat")
test_data_path = os.path.join(cur_dir, "/home/zhanghexu/PJT/DR_Application/ANN_final/data/deleted/Enhenced_Test_Data.mat")

# 检查文件存在
if not os.path.exists(training_data_path):
    raise FileNotFoundError(f"Training data file not found: {training_data_path}")
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test data file not found: {test_data_path}")

# 加载文件
training_data = scipy.io.loadmat(training_data_path)
test_data = scipy.io.loadmat(test_data_path)

# 检查文件内容
required_keys = ["training_samples", "training_labels"]
for key in required_keys:
    if key not in training_data:
        raise ValueError(f"Training data is missing key: {key}")
required_keys = ["test_samples", "test_labels"]
for key in required_keys:
    if key not in test_data:
        raise ValueError(f"Training data is missing key: {key}")
print("文件提取完成")

# 提取数据
# X : [lp @ ln @ hc]^T
# Y : 频率、实部、虚部
print("数据加载阶段开始")
X = training_data["training_samples"]
Y = training_data["training_labels"]
X_test = test_data["test_samples"]
Y_test = test_data["test_labels"]

# 矢量拟合函数
def vector_fitting(Y: np.ndarray, save_path: str = "", verbose: bool = True) -> np.ndarray:
    n_samples = len(Y)
    W = len(Y[0][0])
    samples_vf = []

    for i in range(n_samples):
        S_11 = Y[i][0][:, 1] + (Y[i][0][:, 2] * 1j)
        freqs = Y[i][0][:, 0]

        ntwk = skrf.Network(frequency=freqs, s=S_11, name=f"frequency_response_{i}")
        vf = skrf.VectorFitting(ntwk)
        vf.auto_fit()

        samples_vf.append(vf)

        if verbose:
            model_orders = vf.get_model_order(vf.poles)
            print(f"Sample {i}: Model order = {model_orders}")
            print(f"Real poles = {np.sum(vf.poles.imag == 0.0)}")
            print(f"Complex poles = {np.sum(vf.poles.imag > 0.0)}")
            print(f"RMS Error = {vf.get_rms_error()}\n")

    # 保存拟合结果
    if save_path:
        np.save(save_path, np.array(samples_vf, dtype=object))
        print(f"Vector fitting results saved to {save_path}")

    return samples_vf


# 矢量拟合并保存结果
#save_dir = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/processed_data"
# 删减
save_dir = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/deleted"
# 处理训练集数据
vf_samples_path = os.path.join(save_dir, "vf_samples.npy")
vf_samples = vector_fitting(Y, save_path=vf_samples_path)

# 处理测试集数据
vf_samples_test_path = os.path.join(save_dir, "vf_samples_test.npy")
vf_samples_test = vector_fitting(Y_test, save_path=vf_samples_test_path)

# 加载拟合结果函数
def load_vf_samples(vf_samples_path: str, vf_samples_test_path: str):
    vf_samples = np.load(vf_samples_path, allow_pickle=True)
    vf_samples_test = np.load(vf_samples_test_path, allow_pickle=True)
    return vf_samples, vf_samples_test

