import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.fourier_features as ff


# 设备选择
def get_device():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    return device


def mlp_layers(input_size: int, hidden_size: int, output_size: int):
    layers = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    return layers


class MLP(nn.Module):
    def __init__(self, input_size: int, model_order: int):
        super().__init__()
        self.model_order = model_order
        self.fourier_features = ff.FourierFeatures(input_size, model_order * 2, scale=100)
        self.hidden_size = model_order * 4 + 1
        self.output_size = model_order * 2 + 1
        self.layers = mlp_layers(input_size, self.hidden_size, self.output_size)
        self.optimizer = torch.optim.NAdam(self.parameters(), lr=0.2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.87)

    def forward(self, x):
        x_fourier = F.normalize(x, dim=0)
        pred_coeffs = self.layers(x_fourier)
        const_coeff = pred_coeffs[0]
        complex_coeffs = torch.complex(pred_coeffs[1::2], pred_coeffs[2::2])
        return torch.cat((const_coeff.unsqueeze(0), complex_coeffs), dim=0)


# Example Training Function (customize based on actual use case)
def train_model(model, data_loader, num_epochs):
    device = get_device()
    model.to(device)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            output = model(x)
            loss = loss_fn(output, y)
            epoch_loss += loss.item()

            # Backward and optimize
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

        # Update learning rate
        model.scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {model.optimizer.param_groups[0]['lr']:.6f}")