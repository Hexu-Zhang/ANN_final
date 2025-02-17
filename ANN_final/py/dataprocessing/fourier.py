import torch
import torch.nn as nn
import numpy as np
import unittest
import matplotlib.pyplot as mplt


class FourierFeatures(nn.Module):
    def __init__(self, input_dim, num_features, scale=10):
        super(FourierFeatures, self).__init__()
        torch.manual_seed(17)  # 种子用于复现性
        # 创建矩阵 B
        self.B = nn.Parameter(torch.randn(size=(num_features, input_dim)) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入 x 投影到更高维空间
        x_proj = 2 * np.pi * torch.matmul(x, self.B.T)
        # 连接正弦和余弦变换结果
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class TestFourierFeatures(unittest.TestCase):
    def test_fourier_features_output(self):
        input_dim = 2
        num_features = 10
        scale = 10
        ff = FourierFeatures(input_dim, num_features, scale)

        # Creating test inputs
        test_inputs = torch.tensor([[0.1, 0.2], [10.3, 100.4]])
        outputs = ff(test_inputs)

        # Check the output dimensions
        self.assertEqual(outputs.shape, (2, 2 * num_features))  # Check output size

        # Check output range for sine and cosine values
        self.assertTrue(torch.all((outputs >= -1) & (outputs <= 1)))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

# 对输入数据和测试数据的傅里叶特征变换，并将结果转换为 NumPy 数组
# 可调节参数：features，傅里叶特征的数量过多时会导致过拟合


FF = ff.FourierFeatures(3, 7, scale=0.4).to(device)
X_fourier = FF(tensor_X).detach().cpu().numpy()
X_test_fourier = FF(tensor_X_test).detach().cpu().numpy()
#pca = PCA(n_components=0.95)  # 保留 95% 的方差
#X_pretest_fourier = FF(tensor_X_test).detach().cpu().numpy()
#X_test_fourier = pca.fit_transform(X_pretest_fourier)
print("傅里叶完成")