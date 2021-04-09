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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch and numpy seeds for reproducibility
torch.manual_seed(1)
np.random.seed(1)

# Experiment parameters
rnn_type = 'GRU'
continuous_cols = ['T', 'RH', 'AH']
categorical_cols = []
target_col = 'PT08.S1(CO)'
bidirectional = True
batch_size = 256
input_len = 20
forecast_horizon = 4
input_dim = len(continuous_cols) + len(categorical_cols) + 1
output_dim = 1
hidden_dim = 64
num_layers = 2
lr = 1e-3
epochs = 100

# Load data
df = pd.read_csv(Path('../data/air_quality_simplified.csv'), index_col=['DT'])

# Univariate or multivariate
if input_dim == 1:
    dataset = dataloader.UnivariateTSDataset(df, target_col, input_len, forecast_horizon)
else:
    dataset = dataloader.MultivariateTSDataset(df, target_col, continuous_cols, categorical_cols, input_len,
                                               forecast_horizon)
train_dataset, train_loader, test_dataset, test_loader = dataset.get_loaders(batch_size=batch_size)

# Define models and training objects
if rnn_type == 'LSTM':
    model = rnns.LSTM(input_dim, hidden_dim, output_dim, num_layers, device, bi=bidirectional).to(device)
else:
    model = rnns.GRU(input_dim, hidden_dim, output_dim, num_layers, device, bi=bidirectional).to(device)

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
