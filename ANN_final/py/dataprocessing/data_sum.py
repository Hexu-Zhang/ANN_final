# 启动 ： python3 /home/zhanghexu/PJT/DR_Application/ANN_final/py/dataprocessing/data_sum.py

import os
import numpy as np
import scipy.io

# 设置路径
# 仅增强
#processed_data_dir_new = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/enhenced"
# 带有删除少数项阶
processed_data_dir_new = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/deleted"
# 加载 enhenced_X_train.npy 和 enhenced_Y_train.npy
X_train_path = os.path.join(processed_data_dir_new, "enhenced_X_train.npy")
Y_train_path = os.path.join(processed_data_dir_new, "enhenced_Y_train.npy")
X_test_path = os.path.join(processed_data_dir_new, "enhenced_X_test.npy")
Y_test_path = os.path.join(processed_data_dir_new, "enhenced_Y_test.npy")

try:
    # 加载训练集数据
    X_train = np.load(X_train_path, allow_pickle=True)
    Y_train = np.load(Y_train_path, allow_pickle=True)

    # 加载测试集数据
    X_test = np.load(X_test_path, allow_pickle=True)
    Y_test = np.load(Y_test_path, allow_pickle=True)
except FileNotFoundError:
    print("无法找到训练集或测试集文件，请检查路径。")
    sys.exit()

# 合并训练数据到字典
training_data = {
    "training_samples": X_train,
    "training_labels": Y_train,
}

# 合并测试数据到字典
test_data = {
    "test_samples": X_test,
    "test_labels": Y_test,
}

# 保存为 .mat 文件
mat_file_train = os.path.join(processed_data_dir_new, "Enhenced_Training_Data.mat")
mat_file_test = os.path.join(processed_data_dir_new, "Enhenced_Test_Data.mat")

try:
    # 保存训练数据
    scipy.io.savemat(mat_file_train, training_data)
    # 保存测试数据
    scipy.io.savemat(mat_file_test, test_data)
    print("数据已成功保存为 .mat 文件！")
except Exception as e:
    print(f"保存 .mat 文件时出错：{str(e)}")