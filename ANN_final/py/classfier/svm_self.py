# 启动 python3 /home/zhanghexu/PJT/DR_Application/ANN_final/py/classfier/svm_self.py

import math
import numpy as np

class SVM:
    def __init__(self, C=1.0, gamma=0.1):
        self.C = C
        self.gamma = gamma
        self.support_vectors = None
        self.alphas = None
        self.bias = None

    def rbf_kernel(self, x, x_prime):
        return np.exp(-self.gamma * np.linalg.norm(x - x_prime)**2)

    def train(self, X, Y, iterations=1000, learning_rate=1e-5):
        n_samples, n_features = X.shape
        # 初始化拉格朗日乘子
        self.alphas = np.zeros(n_samples)
        self.bias = 0

        # 随机梯度下降求解拉格朗日乘子
        for _ in tqdm(range(iterations)):
            for i in range(n_samples):
                E = self.predict([X[i]])[0] - Y[i]
                derivative = Y[i] * E
                self.alphas[i] = max(0, min(self.C, self.alphas[i] - learning_rate * derivative))

        # 提取支持向量
        self.support_vectors = X[self.alphas > 1e-7]
        self.support_labels = Y[self.alphas > 1e-7]
        self.support_alphas = self.alphas[self.alphas > 1e-7]

    def predict(self, X):
        n_samples = len(X)
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            kernel_output = np.array([self.rbf_kernel(X[i], x_s) for x_s in self.support_vectors])
            weighted_sum = np.dot(kernel_output, self.support_alphas * self.support_labels)
            y_pred[i] = np.sign(weighted_sum + self.bias)
        return y_pred