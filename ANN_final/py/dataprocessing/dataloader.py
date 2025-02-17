# 启动 ： python3 /home/zhanghexu/PJT/DR_Application/ANN_final/py/dataprocessing/dataloader.py

import os
import sys
import scipy.io
import torch
import numpy as np
import builtins


# 加载 .mat 文件
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

# 设置目标路径
processed_data_dir = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/processed_data"

# 创建目标目录（如果不存在）
os.makedirs(processed_data_dir, exist_ok=True)

# 提取数据
# X : [lp @ ln @ hc]^T
# Y : 频率、实部、虚部
print("数据加载阶段开始")
X = training_data["candidates"]
Y = training_data["responses"]
X_test = test_data["real_test_candidates"]
Y_test = test_data["real_test_responses"]

scipy.io.savemat(os.path.join(processed_data_dir, "X.mat"), {"X": X})
scipy.io.savemat(os.path.join(processed_data_dir, "Y.mat"), {"Y": Y})
scipy.io.savemat(os.path.join(processed_data_dir, "X_test.mat"), {"X_test": X_test})
scipy.io.savemat(os.path.join(processed_data_dir, "Y_test.mat"), {"Y_test": Y_test})

# 检测数据提取情况
print("数据加载完成，开始检测")

# 定义函数 check_and_print_data
def check_and_print_data(X, Y, X_test, Y_test):
    # 检查 X 的形状
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    assert X.shape == (64, 3), f"X 的形状错误，期望是 (64, 3)，实际是 {X.shape}"
    print(f"X 的形状: {X.shape}, 类型: {type(X)}")

    # 检查 Y 的形状
    assert isinstance(Y, np.ndarray) and Y.shape == (64, 1), f"Y 的形状错误，期望是 (64, 1)，实际是 {Y.shape}"
    print(f"Y 的形状: {Y.shape}, 类型: {type(Y)}")
    print("Y 的样本形状:")
    for i in range(Y.shape[0]):
        sample = Y[i, 0]
        assert isinstance(sample, np.ndarray) and sample.shape == (1001, 3), \
            f"Y 的第 {i+1} 个样本形状错误，期望是 (1001, 3)，实际是 {sample.shape}"
        print(f"Y 的第 {i+1} 个样本形状: {sample.shape}")

    # 检查 X_test 的形状
    if not isinstance(X_test, np.ndarray):
        X_test = np.array(X_test)
    assert X_test.shape == (36, 3), f"X_test 的形状错误，期望是 (36, 3)，实际是 {X_test.shape}"
    print(f"X_test 的形状: {X_test.shape}, 类型: {type(X_test)}")

    # 检查 Y_test 的形状
    assert isinstance(Y_test, np.ndarray) and Y_test.shape == (36, 1), f"Y_test 的形状错误，期望是 (36, 1)，实际是 {Y_test.shape}"
    print(f"Y_test 的形状: {Y_test.shape}, 类型: {type(Y_test)}")
    print("Y_test 的样本形状:")
    for i in range(Y_test.shape[0]):
        sample = Y_test[i, 0]
        assert isinstance(sample, np.ndarray) and sample.shape == (1001, 3), \
            f"Y_test 的第 {i+1} 个样本形状错误，期望是 (1001, 3)，实际是 {sample.shape}"
        print(f"Y_test 的第 {i+1} 个样本形状: {sample.shape}")

# 定义函数 save_data_as_npy
def save_data_as_npy(data, filename):
    np.save(filename, data)
    print(f"数据已保存为: {filename}")

# 调用 check_and_print_data 函数
check_and_print_data(X, Y, X_test, Y_test)

print("数据检测及格式输出完成，开始对数据进行归一化操作")
# 归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 打印原始 X 和 X_test 数据
print("\n原始 X 数据：")
for i, row in enumerate(X):
    print(f"样本 {i+1}: {row}")

print("\n原始 X_test 数据：")
for i, row in enumerate(X_test):
    print(f"测试样本 {i+1}: {row}")

# 打印归一化后的 X 和 X_test 数据
print("\n标准化后的 X 数据：")
for i, row in enumerate(X_scaled):
    print(f"标准化后的样本 {i+1}: {row}")

print("\n标准化后的 X_test 数据：")
for i, row in enumerate(X_test_scaled):
    print(f"标准化后的测试样本 {i+1}: {row}")

# 设备选择
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()
tensor_X = torch.FloatTensor(X).to(device)
tensor_X_test = torch.FloatTensor(X_test).to(device)
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# 将 Y 和 Y_test 中的嵌套数据结构提取并转换为标准的 NumPy数组，方便后续的数据处理和模型训练。
Y_data = np.array([y[0] for y in Y])
Y_test_data = np.array([y[0] for y in Y_test])

# 保存 Y_data 和 Y_test_data 为 .npy 文件
save_data_as_npy(Y_data, processed_data_dir + "/Y_data.npy")
save_data_as_npy(Y_test_data, processed_data_dir + "/Y_test_data.npy")

# 分离实部虚部，以便后续使用，Y_data[:, :, 1]是S参数的实部，Y_data[:, :, 2]是S参数的虚部。
freqs = Y_data[:, :, 0]
S_11_real_train = Y_data[:, :, 1]
S_11_imag_train = Y_data[:, :, 2]
S_11_real_test = Y_test_data[:, :, 1]
S_11_imag_test = Y_test_data[:, :, 2]

# 合成复数
S_11_samples_train = S_11_real_train + 1j * S_11_imag_train
S_11_samples_test = S_11_real_test + 1j * S_11_imag_test

# 分离实部虚部，保存为 .npy 文件
save_data_as_npy(S_11_real_train, processed_data_dir + "/S_11_real_train.npy")
save_data_as_npy(S_11_imag_train, processed_data_dir + "/S_11_imag_train.npy")
save_data_as_npy(S_11_real_test, processed_data_dir + "/S_11_real_test.npy")
save_data_as_npy(S_11_imag_test, processed_data_dir + "/S_11_imag_test.npy")

# 合成复数，并保存为 .npy 文件
save_data_as_npy(S_11_samples_train, processed_data_dir + "/S_11_samples_train.npy")
save_data_as_npy(S_11_samples_test, processed_data_dir + "/S_11_samples_test.npy")

# 验证范围
magnitude = np.abs(S_11_samples_train)
print("S11 幅度范围:", np.min(magnitude), np.max(magnitude))
phase = np.angle(S_11_samples_train)
print("S11 相位范围:", np.min(phase), np.max(phase))

# 将 freqs（频率数据）和 S_11_samples_train（训练数据的 S 参数）以及 S_11_samples_test（测试数据的 S 参数）转换为 PyTorch 的张量（Tensor），并将其移动到指定的设备（GPU 或 CPU）上。
tensor_freqs = torch.tensor(freqs, dtype=torch.float32, device=device)
tensor_S_train = torch.tensor(S_11_samples_train, dtype=torch.complex64, device=device)
tensor_S_test = torch.tensor(S_11_samples_test, dtype=torch.complex64, device=device)

# 保存为 .pt 文件
torch.save(tensor_freqs, processed_data_dir + "/tensor_freqs.pt")
torch.save(tensor_S_train, processed_data_dir + "/tensor_S_train.pt")
torch.save(tensor_S_test, processed_data_dir + "/tensor_S_test.pt")
print("\n张量化的频率数据 (tensor_freqs):")
print(tensor_freqs)

print("\n张量化的训练数据 S 参数 (tensor_S_train):")
print(tensor_S_train)

print("\n张量化的测试数据 S 参数 (tensor_S_test):")
print(tensor_S_test)
print("数据张量化完成")