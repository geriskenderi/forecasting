import torch
import rnns
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import dataloader, trainer
from pathlib import Path

# Set up CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch and numpy seeds for reproducibility
torch.manual_seed(1)
np.random.seed(1)

# Experiment parameters
batch_size = 256
input_len = 12
forecast_horizon = 4
input_dim = 1
output_dim = 1
hidden_dim = 32
num_layers = 1
lr = 1e-3
epochs = 100

# Load data
df = pd.read_csv(Path('../data/air_quality_simplified.csv'), index_col=['DT'])
univariate_dataset = dataloader.UnivariateTSDataset(df, "PT08.S1(CO)", input_len, forecast_horizon)
train_dataset, train_loader, test_dataset, test_loader = univariate_dataset.get_loaders(batch_size=batch_size)

# Define models and training objects
model = rnns.LSTM(input_dim, hidden_dim, output_dim, num_layers, device).to(device)
# model = rnns.GRU(input_dim, hidden_dim, output_dim, num_layers, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
loss_fn = nn.MSELoss()

# Train and evaluate model
trn = trainer.Trainer(model, train_loader, test_loader, loss_fn, optimizer, device, lr_scheduler)
trn.train(epochs)

# Make predictions on testing set
gt, forecasts = trn.forecast()
mae = mean_absolute_error(gt, forecasts)
rmse = mean_squared_error(gt, forecasts, squared=False)
print('\nMAE: {}, RMSE: {}'.format(round(mae, 4), round(rmse, 4)))
