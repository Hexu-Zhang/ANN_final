import argparse
import os
import numpy as np
import scipy.io  # Read Matlab files
from sklearn.decomposition import PCA
from sklearn import svm  # SVM
from sklearn.metrics import mean_absolute_percentage_error, classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from tqdm import trange, tqdm
import joblib
import matplotlib.pyplot as mplt
import torch
from utils import *
import models.fourier_features as ff
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import pickle


# 初始化阶段：数据设定、设备加载、主函数选择
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

# 数据加载阶段：加载数据，进行归一化等预处理
print("文件提取阶段开始")

# 获取路径
cur_dir = os.path.dirname(os.path.realpath(__file__))
# 加载对应路径的文件
training_data_path = os.path.join(cur_dir, "data/Training_Data.mat")
test_data_path = os.path.join(cur_dir, "data/Real_Test_Data.mat")
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
print("数据加载阶段开始")
X = training_data["candidates"]
Y = training_data["responses"]
X_test = test_data["real_test_candidates"]
Y_test = test_data["real_test_responses"]
# X = [lp @ ln @ hc]^T (meters), of shape (64, 3) here.
# Y is S_11 over the frequency range (GHz) with 3 vals per sample representing: [frequency (GHz), real, imaginary]

# 检测数据提取情况
print("数据加载完成，开始检测")
check_and_print_data(X, Y, X_test, Y_test)

print("数据检测及格式输出完成，开始对数据进行归一化操作")
# 归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 打印原始 X 和 X_test 数据
#print("\n原始 X 数据：")
#for i, row in enumerate(X):
#    print(f"样本 {i+1}: {row}")

#print("\n原始 X_test 数据：")
#for i, row in enumerate(X_test):
#    print(f"测试样本 {i+1}: {row}")

# 打印归一化后的 X 和 X_test 数据
#print("\n标准化后的 X 数据：")
#for i, row in enumerate(X_scaled):
#    print(f"标准化后的样本 {i+1}: {row}")

#print("\n标准化后的 X_test 数据：")
#for i, row in enumerate(X_test_scaled):
#    print(f"标准化后的测试样本 {i+1}: {row}")

# 设备选择
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
save_data_as_npy(Y_data, "Y_data.npy")
save_data_as_npy(Y_test_data, "Y_test_data.npy")

# 分离实部虚部，以便后续使用，Y_data[:, :, 1]是S参数的实部，Y_data[:, :, 2]是S参数的虚部。
freqs = Y_data[:, :, 0]
S_11_real_train = Y_data[:, :, 1]
S_11_imag_train = Y_data[:, :, 2]
S_11_real_test = Y_test_data[:, :, 1]
S_11_imag_test = Y_test_data[:, :, 2]

# 合成复数
S_11_samples_train = S_11_real_train + 1j * S_11_imag_train
S_11_samples_test = S_11_real_test + 1j * S_11_imag_test

# 保存分离后的实部、虚部
save_data_as_npy(S_11_real_train, "S_11_real_train.npy")
save_data_as_npy(S_11_imag_train, "S_11_imag_train.npy")
save_data_as_npy(S_11_real_test, "S_11_real_test.npy")
save_data_as_npy(S_11_imag_test, "S_11_imag_test.npy")

# 保存合成后的复数
save_data_as_npy(S_11_samples_train, "S_11_samples_train.npy")
save_data_as_npy(S_11_samples_test, "S_11_samples_test.npy")

#验证范围
magnitude = np.abs(S_11_samples_train)
#print("S11 幅度范围:", np.min(magnitude), np.max(magnitude))
phase = np.angle(S_11_samples_train)
#print("S11 相位范围:", np.min(phase), np.max(phase))

# 将 freqs（频率数据）和 S_11_samples_train（训练数据的 S 参数）以及 S_11_samples_test（测试数据的 S 参数）转换为 PyTorch 的张量（Tensor），并将其移动到指定的设备（GPU 或 CPU）上。
tensor_freqs = torch.tensor(freqs, dtype=torch.float32, device=device)
tensor_S_train = torch.tensor(S_11_samples_train, dtype=torch.complex64, device=device)
tensor_S_test = torch.tensor(S_11_samples_test, dtype=torch.complex64, device=device)
torch.save(tensor_freqs, "tensor_freqs.pt")
torch.save(tensor_S_train, "tensor_S_train.pt")
torch.save(tensor_S_test, "tensor_S_test.pt")
#print("\n张量化的频率数据 (tensor_freqs):")
#print(tensor_freqs)

#print("\n张量化的训练数据 S 参数 (tensor_S_train):")
#print(tensor_S_train)

#print("\n张量化的测试数据 S 参数 (tensor_S_test):")
#print(tensor_S_test)
print("数据张量化完成")

# 对输入数据和测试数据的傅里叶特征变换，并将结果转换为 NumPy 数组
# 可调节参数：features，傅里叶特征的数量过多时会导致过拟合
# 可调节参数：缩放因子scale，影响频率范围，过大导致过拟合
# 是否需要导出分析？ 暂时先不改？
FF = ff.FourierFeatures(3, 7, scale=0.4).to(device)
X_fourier = FF(tensor_X).detach().cpu().numpy()
X_test_fourier = FF(tensor_X_test).detach().cpu().numpy()
#pca = PCA(n_components=0.95)  # 保留 95% 的方差
#X_pretest_fourier = FF(tensor_X_test).detach().cpu().numpy()
#X_test_fourier = pca.fit_transform(X_pretest_fourier)
print("傅里叶完成")

# 主函数选择阶段：三个状态的选择，最终进行对应的任务
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains or evaluates an ANN to predict the S_11 freq response using .mat files")
    parser.add_argument("--train", action="store_true",
                        help="Train ANNs from the matlab data files. (else will load files if present).")
    parser.add_argument("--finetune", action="store_true",
                        help="Load and continue to Train ANNs from saved files from a previous --train run.")
    parser.add_argument("--plot", action="store_true", help="Plot data samples with matplotlib.")
    args = parser.parse_args()



    if args.train:
        print("重新训练开始.")
        # 进行SVM训练以及ANN训练

        # SVM训练
        # 原代码中，SVM始终存在过拟合现象。综合考虑
        # 论文中使用SVM，是想将原有的参数按照传递函数的阶数进行分类，因此使用几何变量和对应的TF阶数作为输入和输出，训练SVM模型，以便在测试过程中对几何变量进行分类。
        # 几何变量：输入到SVM的特征向量 x，
        # 首先使用矢量拟合，从EM相应中提取TF的极点和留数
        print("SVM训练开始")

        ## 向量拟合
        #vf_samples = vector_fitting(Y)
        #vf_samples_test = vector_fitting(Y_test)

        # 计算每个样本的模型阶数（model order），并分别存储训练数据和测试数据的模型阶数。
        # 调用测试发现，模型阶数是一个实数、2n个复数，但是得到的是直接相加，复数极点没有乘二
        # 经过测试，并不影响

        #model_orders_observed = [len(vf.poles) for vf in vf_samples]
        #model_orders_test_observed = [len(vf.poles) for vf in vf_samples_test]
        #model_orders_observed = [x * 2 - 1 for x in model_orders_preobserved]
        #model_orders_test_observed = [x * 2 - 1 for x in model_orders_test_preobserved]

        #print("X_fourier:")
        #print(X_fourier)
        #print("model_orders_observed:")
        #print(model_orders_observed)

        #with open('vf_samples.pkl', 'wb') as f:
        #    pickle.dump(vf_samples, f)

        #with open('vf_samples_test.pkl', 'wb') as f:
        #    pickle.dump(vf_samples_test, f)
        #np.save('model_orders_observed.npy', model_orders_observed)
        #np.save('model_orders_test_observed.npy', model_orders_test_observed)
        with open('vf_samples.pkl', 'rb') as f:
            vf_samples = pickle.load(f)

        with open('vf_samples_test.pkl', 'rb') as f:
            vf_samples_test = pickle.load(f)
        model_orders_observed = np.load('model_orders_observed.npy')
        model_orders_test_observed = np.load('model_orders_test_observed.npy')

        # Train SVM:
        # Need to predict the Order based on the input S-parameter (over frequency space).
        # tried over ['linear', 'poly', 'rbf', sigmoid']. ovo vs ovr doesn't seem to matter.
        # svc = svm.SVC(kernel='rbf') # Use this class if not using "balanced" or FourierFeatures.
        # Use probability (otherwise it doesn't calculate their cross validation) and set the random state seed.
        # 使用的工具箱，先跑起来，后续试试能不能自己写

        # 初始化 SVM 模型
        svc = SVC(
            kernel='poly',
            degree=4,
            gamma='scale',
            C=0.5,
            class_weight="balanced",
            probability=True,
            random_state=1
        )


        # 封装 SVM 模型到管道
        clf = make_pipeline(StandardScaler(), svc)
        clf.fit(X_fourier, model_orders_observed)
        # clf.fit(X, model_orders_observed)



        # 保存模型
        print("SVM has been fit, saving to pickle.")
        joblib.dump(clf, "model_weights_output/svm.pkl")

        # SVM predict on Train Data for a sanity check.
        print(f"Train Actual Model Orders (VF): {model_orders_observed}")
        model_orders_predicted = clf.predict(X_fourier)
        # model_orders_predicted = clf.predict(X)
        print(f"Train Predicted Model Orders: {model_orders_predicted}")

        # SVM predict on Test Data
        print(f"Test Actual (VF) Model Orders: {model_orders_observed}")
        model_orders_test_predicted = clf.predict(X_test_fourier)
        # model_orders_test_predicted = clf.predict(X_test)
        print(f"Test Predicted Model Orders: {model_orders_test_predicted}")

        # Evaluate Average training Accuracy
        train_accuracy = accuracy_score(model_orders_observed, model_orders_predicted)
        train_conf_matrix = confusion_matrix(model_orders_observed, model_orders_predicted)
        train_cls_report = classification_report(model_orders_observed, model_orders_predicted, zero_division=0)
        print(f"Training SVM Accuracy is: {train_accuracy * 100}%")
        #print(f"Training SVM Confusion Matrix\n", train_conf_matrix)
        #print(f"Training SVM Classification Report\n", train_cls_report)

        # Evaluate Average testing Accuracy
        test_accuracy = accuracy_score(model_orders_test_observed, model_orders_test_predicted)
        test_conf_matrix = confusion_matrix(model_orders_test_observed, model_orders_test_predicted)
        test_cls_report = classification_report(model_orders_test_observed, model_orders_test_predicted,
                                                zero_division=0)
        print(f"Testing SVM Accuracy is: {test_accuracy * 100}%")
        #print(f"Testing SVM Confusion Matrix\n", test_conf_matrix)
        #print(f"Testing SVM Classification Report\n", test_cls_report)


        ## Train ANN on EM simulation results and Outputs of pole-residue-based transfer function: ##
        print(f"ANN训练开始")

        # 创建ANN模型
        ANNs = create_neural_models(vf_samples, tensor_X, tensor_S_train, tensor_freqs, plot=args.plot, epochs=0)
        print("Pre-training on vector-fitting coefficients finished. Beginning fine-tuning with training data.")

        # ANN模型训练
        train_neural_models(ANNs, model_orders_predicted, tensor_X, tensor_S_train, tensor_freqs, epochs=36)

        print("Training finished, saving models.")
        for order, models in ANNs.items():
            torch.save(models[0], f"model_weights_output/s_param_ann_order_{order}_p.pkl")
            torch.save(models[1], f"model_weights_output/s_param_ann_order_{order}_r.pkl")

    else:
        # Else load pre trained models.
        print("Initializing testing environment. Loading weights files.")

        print("Loading SVM file pickle.")
        clf = joblib.load("model_weights_output/svm.pkl")

        # SVM predict on Train and Test data
        model_orders_predicted = clf.predict(X_fourier)
        model_orders_test_predicted = clf.predict(X_test_fourier)
        print(f"Train Predicted: {model_orders_predicted}")
        print(f"Test Predicted: {model_orders_test_predicted}")

        ANNs = {}
        for order in set(np.concatenate([model_orders_predicted, model_orders_test_predicted])):
            models = [torch.load(f"model_weights_output/s_param_ann_order_{order}_p.pkl", map_location=device),
                      torch.load(f"model_weights_output/s_param_ann_order_{order}_r.pkl", map_location=device)]
            ANNs[order] = models

    # 精校
    if args.finetune:
        print("Models loaded, 'finetune' selected. Training more.")
        for order, models in ANNs.items():
            for model in models:
                model.optimizer = torch.optim.SGD(model.parameters(), lr=0.000002, momentum=0.9)
        train_neural_models(ANNs, model_orders_predicted, tensor_X, tensor_S_train, tensor_freqs, epochs=3)

        save_models = input("Training finished, save models? Y/n: ")
        if save_models.lower() == "y":
            print("Saving models.")
            for order, models in ANNs.items():
                torch.save(models[0], f"model_weights_output/s_param_ann_order_{order}_p.pkl")
                torch.save(models[1], f"model_weights_output/s_param_ann_order_{order}_r.pkl")
        else:
            print("Models not saved.")

    print(f"Now beginning inference.")
    # Sanity check with Training data
    print("Starting sample run on training (if loss not low, model failed to fit)")
    S_predicted_samples_train, train_loss_avg = predict_samples(ANNs, model_orders_predicted, tensor_X, tensor_S_train,
                                                                tensor_freqs)
    print("Average training MAPE:", train_loss_avg * 100)

    # Test data
    print("Starting test run for actual accuracy.")
    S_predicted_samples_test, test_loss_avg = predict_samples(ANNs, model_orders_test_predicted, tensor_X_test,
                                                              tensor_S_test, tensor_freqs)
    print("Average testing MAPE:", test_loss_avg * 100)

    # Y: 训练数据的响应
    # Y_test: 测试数据的响应
    # S_11_samples_train: 训练数据的 S11 样本
    # S_predicted_samples_train: 训练数据的预测 S11 样本
    # S_11_samples_test: 测试数据的 S11 样本
    # S_predicted_samples_test: 测试数据的预测 S11 样本
    # freqs: 频率数据
    # model_orders_predicted: 训练数据的模型阶数
    # model_orders_test_predicted: 测试数据的模型阶数

    # Plot neural net predictions
    # if args.plot:
    if True:
        print("Plotting Train data")
        for i in range(len(Y)):
            mplt.plot(freqs[i], 20 * np.log10(np.abs(S_11_samples_train[i])), 'r-', label="Source (HFSS)")
            mplt.plot(freqs[i], 20 * np.log10(np.abs(S_predicted_samples_train[i].cpu().detach().numpy())), 'b-.',
                      label="ANN")
            mplt.xlabel("Frequency (GHz)")
            mplt.ylabel("S_11 (dB)")
            mplt.title(f"Order {model_orders_predicted[i]}")
            mplt.legend()
            mplt.savefig(f"plots/train_plot_{i}.png")
            mplt.clf()
        print("Plotting Test data")
        for i in range(len(Y_test)):
            mplt.plot(freqs[i], 20 * np.log10(np.abs(S_11_samples_test[i])), 'r-')
            mplt.plot(freqs[i], 20 * np.log10(np.abs(S_predicted_samples_test[i].cpu().detach().numpy())), 'b-.')
            mplt.xlabel("Frequency (GHz)")
            mplt.ylabel("S_11 (dB)")
            mplt.title(f"Order {model_orders_test_predicted[i]}")
            mplt.savefig(f"plots/test_plot_{i}.png")
            mplt.clf()