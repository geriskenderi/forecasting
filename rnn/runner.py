import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
import statsmodels.api as sm
from utils import data, dataloader
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set up CUDA
# lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# torch and numpy seeds for reproducibility
torch.manual_seed(1)
np.random.seed(1)

# Experiment parameters
batch_size = 1
input_len = 7
forecast_horizon = 1

# Load data
df = pd.read_csv(Path('../data/air_quality_simplified.csv'), index_col=['DT'])
univariate_dataset = dataloader.UnivariateTSDataset(df, "PT08.S1(CO)", input_len, forecast_horizon)
train_loader, test_loader = univariate_dataset.get_loaders(batch_size=batch_size)
print()
