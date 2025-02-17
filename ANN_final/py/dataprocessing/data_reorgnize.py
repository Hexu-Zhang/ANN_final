# 启动 python3 /home/zhanghexu/PJT/DR_Application/ANN_final/py/dataprocessing/data_reorgnize.py

import os
import sys
import scipy.io
import torch
import numpy as np
import builtins

from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 数据标准化工具
from builtins import ValueError, FileNotFoundError  # 内置异常类

# 获取路径
cur_dir = os.path.dirname(os.path.realpath(__file__))

# 加载对应路径的文件
training_data_path = os.path.join(cur_dir, "/home/zhanghexu/PJT/DR_Application/ANN_final/data/original/Training_Data.mat")
test_data_path = os.path.join(cur_dir, "/home/zhanghexu/PJT/DR_Application/ANN_final/data/original/Real_Test_Data.mat")

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
X = training_data["candidates"]
Y = training_data["responses"]
X_test = test_data["real_test_candidates"]
Y_test = test_data["real_test_responses"]

# 定义分组信息
rank_groups = {
    11: [[41, 63], None],
    10: [[1, 2, 6, 12, 15, 20, 24, 26, 29, 30, 48, 52], [7, 11, 20, 31]],
    9: [[8, 10, 13, 17, 22, 23, 31, 33, 34, 35, 37, 39, 46, 49, 51, 53, 55, 58, 60, 61], [12, 13, 21, 22, 24, 26, 27, 29, 30, 33, 36]],
    8: [[3, 4, 5, 7, 9, 14, 16, 18, 19, 25, 27, 28, 32, 36, 38, 40, 42, 43, 44, 45, 47, 50, 54, 56, 57, 59, 62, 64], [1, 2, 6, 8, 9, 10, 14, 16, 17, 18, 19, 23, 25, 28, 32, 34, 35]],
    7: [[11, 21], [4, 15]],
    6: [None, [3]]
}

# 将 MatLab 的 1-based 索引转换为 Python 的 0-based 索引
for rank in rank_groups:
    train_indices, test_indices = rank_groups[rank]
    if train_indices is not None and isinstance(train_indices, list):
        rank_groups[rank][0] = [idx - 1 for idx in train_indices]
    if test_indices is not None and isinstance(test_indices, list):
        rank_groups[rank][1] = [idx - 1 for idx in test_indices]

# 定义分片函数
def split_data(X, Y, X_test, Y_test, rank_groups):
    split_result = {}
    for rank, (train_indices, test_indices) in rank_groups.items():
        # 获取训练集和测试集的 X 和 Y
        if train_indices is not None:
            X_train = X[train_indices]
            Y_train = Y[train_indices]
        else:
            X_train = np.array([])  # 如果没有训练集索引，则直接设置为空数组
            Y_train = np.array([])

        if test_indices is not None:
            X_test_split = X_test[test_indices]
            Y_test_split = Y_test[test_indices]
        else:
            X_test_split = np.array([])  # 如果没有测试集索引，则直接设置为空数组
            Y_test_split = np.array([])

        split_result[rank] = {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_test": X_test_split,
            "Y_test": Y_test_split
        }

        # 检查索引是否超出范围
        if test_indices is not None and len(test_indices) > 0:
            if max(test_indices) >= len(X_test):
                raise IndexError(f"Test index {max(test_indices)} out of bounds for X_test (size {len(X_test)})")

    return split_result

# 数据分片
split_result = split_data(X, Y, X_test, Y_test, rank_groups)

# 打印分片结果
print("分片结果：")
for rank, data in split_result.items():
    print(f"阶数: {rank}")
    print(f"X_train 形状: {data['X_train'].shape}, X_test 形状: {data['X_test'].shape}")
    print(f"Y_train 形状: {data['Y_train'].shape}, Y_test 形状: {data['Y_test'].shape}")
    print('-' * 40)

# 设置新的数据保存路径
processed_data_dir_new = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/enhenced"
os.makedirs(processed_data_dir_new, exist_ok=True)

# 定义分片数据的保存逻辑
increase_dict = {10: 25, 6:15,7:15,11:15}  # 阶数对应的扩充目标数

for rank, data in split_result.items():
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    # 合并 X_train 和 X_test
    if len(X_train) > 0 and len(X_test) > 0:
        X_combined = np.concatenate((X_train, X_test), axis=0)
    elif len(X_train) > 0:
        X_combined = X_train
    elif len(X_test) > 0:
        X_combined = X_test
    else:
        X_combined = np.array([])

    # 合并 Y_train 和 Y_test
    if len(Y_train) > 0 and len(Y_test) > 0:
        Y_combined = np.concatenate((Y_train, Y_test), axis=0)
    elif len(Y_train) > 0:
        Y_combined = Y_train
    elif len(Y_test) > 0:
        Y_combined = Y_test
    else:
        Y_combined = np.array([])

    # 过采样：根据特定的阶数设定目标数目
    if X_combined.size > 0 and Y_combined.size > 0:
        # 获取当前阶数对应的扩充目标数目
        if rank in increase_dict:
            target_samples = increase_dict[rank]
        else:
            target_samples = 20  # 默认目标数目

        if X_combined.shape[0] < target_samples:
            num_samples_needed = target_samples - X_combined.shape[0]
            indices = np.random.choice(X_combined.shape[0], size=num_samples_needed, replace=True)
            X_combined = np.concatenate([X_combined, X_combined[indices]])
            Y_combined = np.concatenate([Y_combined, Y_combined[indices]])

        # 重新划分训练集和测试集（4:1）
        X_train_new, X_test_new, Y_train_new, Y_test_new = train_test_split(
            X_combined,
            Y_combined,
            test_size=0.2,
            random_state=42
        )

    # 保存到文件
    if X_combined.size > 0:
        np.save(os.path.join(processed_data_dir_new, f"X_order_{rank}.npy"), X_combined)
    if Y_combined.size > 0:
        np.save(os.path.join(processed_data_dir_new, f"Y_order_{rank}.npy"), Y_combined)

# 合并所有训练集和测试集
all_X_train = []
all_Y_train = []
all_X_test = []
all_Y_test = []

for rank, data in split_result.items():
    X_combined = np.load(os.path.join(processed_data_dir_new, f"X_order_{rank}.npy"), allow_pickle=True)
    Y_combined = np.load(os.path.join(processed_data_dir_new, f"Y_order_{rank}.npy"), allow_pickle=True)

    # 重新划分训练集和测试集（4:1）
    X_train_new, X_test_new, Y_train_new, Y_test_new = train_test_split(
        X_combined,
        Y_combined,
        test_size=0.2,
        random_state=42
    )

    all_X_train.append(X_train_new)
    all_Y_train.append(Y_train_new)
    all_X_test.append(X_test_new)
    all_Y_test.append(Y_test_new)

# 合并所有阶数的训练集和测试集
if all_X_train and all_Y_train:
    enhenced_X_train = np.concatenate(all_X_train, axis=0)
    enhenced_Y_train = np.concatenate(all_Y_train, axis=0)
    np.save(os.path.join(processed_data_dir_new, "enhenced_X_train.npy"), enhenced_X_train)
    np.save(os.path.join(processed_data_dir_new, "enhenced_Y_train.npy"), enhenced_Y_train)

if all_X_test and all_Y_test:
    enhenced_X_test = np.concatenate(all_X_test, axis=0)
    enhenced_Y_test = np.concatenate(all_Y_test, axis=0)
    np.save(os.path.join(processed_data_dir_new, "enhenced_X_test.npy"), enhenced_X_test)
    np.save(os.path.join(processed_data_dir_new, "enhenced_Y_test.npy"), enhenced_Y_test)

print("数据处理完成！")