import skrf
import torch
import numpy as np
import matplotlib.pyplot as mplt

from tqdm import trange, tqdm
from torchmetrics.functional.regression import mean_absolute_percentage_error as torch_mean_absolute_percentage_error
from models.mlp import MLP, get_device

# H(s) = Sigma(r_i / (s - p_i)) from i=1 to Q (Q is the order of the TF)
def PoleResidueTF(d: float, e: float, poles: torch.Tensor, residues: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-9
    device = get_device()
    # H is freq response
    H = torch.zeros(len(freqs), dtype=torch.cdouble).to(device)
    # s is the angular complex frequency
    s = 2j * np.pi * freqs
    for j in range(len(s)):
        # import pdb; pdb.set_trace()
        H[j] += d + s[j] * e
        for i in range(len(poles)):
            p = poles[i]
            r = residues[i]
            denominator = (s[j] - p)
            if torch.abs(denominator) < epsilon:
                print("Warning, pole inf detected due to pole singularity.")
                denominator += epsilon * (1 if denominator.real >= 0 else -1)
            if torch.imag(p) == 0:
                H[j] += r / denominator
            else:
                H[j] += r / denominator + torch.conj(r) / (s[j] - torch.conj(p))
    return H


# Used for model evaluation
# Using the loss given in eq 10 of (Ding et al.: Neural-Network Approaches to EM-Based Modeling of Passive Components)
def loss_fn(actual_S: torch.Tensor, predicted_S: torch.Tensor) -> float:
    # Take complex conjugate to do square
    c = predicted_S - actual_S
    return torch.sum(torch.abs(c * torch.conj(c))) / 2
    # return torch.sum(torch.abs(torch.pow(c, 2))) / 2


def error_mape(actual_S: torch.Tensor, predicted_S: torch.Tensor) -> float:
    # S is complex, but MAPE isn't. Do average of r and i parts.
    real_MAPE = torch_mean_absolute_percentage_error(actual_S.real, predicted_S.real)
    imag_MAPE = torch_mean_absolute_percentage_error(actual_S.imag, predicted_S.imag)
    return (real_MAPE + imag_MAPE) / 2


def predict(p_model, r_model, input_X: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    if p_model.training:
        pred_e_poles = p_model.forward(input_X)
        pred_d_residues = r_model.forward(input_X)
        pred_e, pred_poles = pred_e_poles[0], pred_e_poles[1:]
        pred_d, pred_residues = pred_d_residues[0], pred_d_residues[1:]
        # print("ANN Poles", pred_poles.detach().cpu().numpy())
        # print("ANN Residues", pred_residues.detach().cpu().numpy())
        return pred_d, pred_e, pred_poles, pred_residues
    else:
        # Don't calculate gradients in eval mode
        with torch.no_grad():
            pred_e_poles = p_model.forward(input_X)
            pred_d_residues = r_model.forward(input_X)
            pred_e, pred_poles = pred_e_poles[0], pred_e_poles[1:]
            pred_d, pred_residues = pred_d_residues[0], pred_d_residues[1:]
            return pred_d, pred_e, pred_poles, pred_residues


def create_neural_models(vf_series: list, X: torch.Tensor, Y: torch.Tensor, freqs: torch.Tensor, epochs: int = 3,
                         plot: bool = False) -> dict:
    x_dims = len(X[0])
    model_orders = [len(vf.poles) for vf in vf_series]
    order_set = set(model_orders)
    ANNs = {}

    device = get_device()

    for order in order_set:
        models = [MLP(x_dims, order).to(device), MLP(x_dims, order).to(device)]
        ANNs[order] = models

    for epoch in range(0, epochs):
        print(f"Starting Epoch {epoch}")
        current_loss = 0.0

        # 使用 tqdm 包装循环，添加进度条
        with tqdm(total=len(model_orders), desc=f"Epoch {epoch} Progress") as pbar:
            for i in range(len(model_orders)):
                model_order = model_orders[i]
                models = ANNs[model_order]
                [model.optimizer.zero_grad() for model in models]

                vf_d = vf_series[i].constant_coeff.item()
                vf_e = vf_series[i].proportional_coeff.item()
                vf_poles = torch.from_numpy(vf_series[i].poles).to(device)
                vf_residues = torch.from_numpy(vf_series[i].residues[0]).to(device)
                vf_S = PoleResidueTF(vf_d, vf_e, vf_poles, vf_residues, freqs[i])

                pred_d, pred_e, pred_poles, pred_residues = predict(models[0], models[1], X[i], freqs[i])
                pred_S = PoleResidueTF(pred_d, pred_e, pred_poles, pred_residues, freqs[i])

                if plot and i == 0:  # 在第一个样本时显示 plot（可选）
                    print(f"SAMPLE {i} of ORDER {model_order}")
                    # 其他打印内容

                loss = 10000 * (torch.norm(vf_d - pred_d, p=2) +
                                torch.norm(vf_e - pred_e, p=2) +
                                torch.norm(vf_poles - pred_poles, p=2) +
                                torch.norm(vf_residues - pred_residues, p=2))
                loss.backward()
                [model.optimizer.step() for model in models]
                current_loss += loss.item()

                if i != 0 and i % 10 == 0:
                    print(f"Loss after mini-batch (model order {model_order}) %5d: %.3f" % (i, current_loss / 500))
                    current_loss = 0.0

                # 更新进度条
                pbar.update(1)

        for _, models in ANNs.items():
            [model.scheduler.step() for model in models]

    for _, models in ANNs.items():
        [model.eval() for model in models]

    return ANNs


# X is the geometrical input to the model.
# Y is only used for training after the predicted coefficients are plugged in.
def train_neural_models(ANNs: dict, model_orders: np.ndarray, X: torch.Tensor, Y: torch.Tensor, freqs: torch.Tensor,
                        epochs=15):
    device = get_device()
    # 设置模型为训练模式
    for _, models in ANNs.items():
        [model.train() for model in models]

    # 使用 tqdm 包裹 epoch 循环以显示进度条
    for epoch in tqdm(range(epochs), desc="Epochs Progress", total=epochs):
        current_loss = 0.0
        # 使用 tqdm 包裹样本循环以显示进度条
        for i in tqdm(range(len(X)), desc=f"Epoch {epoch} Progress", total=len(X), leave=False):
            model_order = model_orders[i]
            models = ANNs[model_order]

            # 梯度置零
            [model.optimizer.zero_grad() for model in models]

            # 获取真实数据
            S_11 = Y[i]

            # 使用 ANN 预测系数
            pred_d, pred_e, pred_poles, pred_residues = predict(models[0], models[1], X[i], freqs[i])
            pred_S = PoleResidueTF(pred_d, pred_e, pred_poles, pred_residues, freqs[i])

            # 计算损失
            loss = loss_fn(S_11, pred_S)
            loss.backward()
            [model.optimizer.step() for model in models]
            current_loss += loss.item()

            # 每 10 个样本打印一次损失，同时更新进度条后的附加信息
            if i != 0 and i % 10 == 0:
                tqdm.write(f"Loss after mini-batch (model order {model_order}) %5d: %.3f" % (i, current_loss / 500))
                current_loss = 0.0

        # 更新学习率调度器
        for _, models in ANNs.items():
            [model.scheduler.step() for model in models]

    # 设置模型为评估模式
    for model_order, models in ANNs.items():
        [model.eval() for model in models]


def predict_samples(ANNs: dict, model_orders: np.ndarray, X: torch.Tensor, Y: torch.Tensor, freqs: torch.Tensor) -> \
tuple[list, float]:
    device = get_device()
    # Filter based on test observation
    # Get order for each sample.
    S_predicted_samples = []
    S_predicted_mape_avg = 0.0
    for i in range(len(model_orders)):
        S_11 = Y[i]
        model_order = model_orders[i]
        models = ANNs[model_order]

        # Predict S_11
        pred_d, pred_e, pred_poles, pred_residues = predict(models[0], models[1], X[i], freqs[i])
        pred_S = PoleResidueTF(pred_d, pred_e, pred_poles, pred_residues, freqs[i])
        S_predicted_samples.append(pred_S)

        # Calculate Loss
        loss = loss_fn(S_11, pred_S)
        S_predicted_mape_avg += error_mape(S_11, pred_S).item()
        if i % 10 == 0:
            print(f"Loss of prediction {i}: {loss.item()}")
    S_predicted_mape_avg /= len(model_orders)
    return S_predicted_samples, S_predicted_mape_avg
