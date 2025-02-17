# 启动：python3 /home/zhanghexu/PJT/DR_Application/ANN_final/py/LogisticRegression.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def perform_multiclass_logistic_regression_with_weights(test_file, target_col, weights_file):
    """
    使用预训练的权重文件进行多分类逻辑回归预测。

    参数:
        test_file (str): 测试集文件路径，文件应为CSV格式。
        target_col (str): 目标变量的列名。
        weights_file (str): 权重文件路径，应为.npy格式。
    """
    # 加载测试数据
    data_test = pd.read_csv(test_file)
    X_test = data_test.drop(columns=[target_col])
    y_test = data_test[target_col]

    # 标准化特征数据
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    # 加载预训练的权重和偏置
    weights_dict = np.load(weights_file, allow_pickle=True).item()

    # 提取权重和偏置，并确保类别键为整数
    weights = {int(k): v['w'] for k, v in weights_dict.items()}
    biases = {int(k): v['bias'] for k, v in weights_dict.items()}
    classes = list(weights.keys())

    # 预测函数
    def predict(X, weights, biases):
        scores = []
        for cls in classes:
            score = np.dot(X, weights[cls]) + biases[cls]
            scores.append(score)
        scores = np.array(scores).T
        return [classes[i] for i in np.argmax(scores, axis=1)]

    # 预测测试集
    y_pred_test = predict(X_test_scaled, weights, biases)

    # 计算准确率
    accuracy = np.mean(y_pred_test == y_test)
    print("测试集准确率：", accuracy)

# 使用指定的文件路径和权重文件
test_file = "/home/zhanghexu/PJT/DR_Application/ANN_final/data/test.csv"
target_col = "orders"
weights_file = "/home/zhanghexu/PJT/DR_Application/ANN_final/logistic_regression_weights.npy"  # 替换为实际的权重文件路径

perform_multiclass_logistic_regression_with_weights(test_file, target_col, weights_file)