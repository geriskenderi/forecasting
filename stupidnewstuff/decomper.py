import torch
import torch.nn as nn
import pytorch_lightning as pl
from stupidnewstuff.dtw import SoftDTWLoss 
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Decomper(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device, bi=False, dropout=0.1):
        # Instantiate variables
        super(Decomper, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.bidirectional = bi

        # Layers
        in_fc = hidden_dim * 2 if bi else hidden_dim
        self.trend_encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=bi)
        self.seasonality_encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=bi)
        self.residual_encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=bi)
        self.trend_out = nn.Linear(in_fc, output_dim)
        self.seasonality_out = nn.Linear(in_fc, output_dim)
        self.residual_out = nn.Sequential(
            nn.Linear(in_fc, output_dim),
            nn.ReLU()
        )

        # Loss functions
        self.val_loss = nn.MSELoss()
        self.shape_loss = SoftDTWLoss()

    def forward(self, x):
        trend_, _ = self.trend_encoder(x)
        seasonality_, _ = self.seasonality_encoder(x)
        residual_, _ = self.residual_encoder(x)

        # Take the wanted future steps through the linear layer
        trend = self.trend_out(trend_)
        seasonality = self.seasonality_out(seasonality_)
        residual = self.residual_out(residual_)

        reconstruction = trend + seasonality + residual

        return reconstruction, (trend, seasonality, residual)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
    
        return [optimizer]


    def training_step(self, train_batch, batch_idx):
        x = train_batch 
        reconstruction, (trend, seasonality, residual) = self.forward(x)
        log_ts, log_reconstruction = torch.log(x), torch.log(reconstruction)
        l1 = self.val_loss(log_ts, log_reconstruction)
        l2 = self.shape_loss(log_ts, log_reconstruction)
        loss = l1 + l2
        self.log('train_loss', loss)

        return loss

    def on_validation_start(self):
        self.reconstructions = []
        self.ground_truth = []

    def validation_step(self, test_batch, batch_idx):
        x = test_batch 
        reconstruction, (trend, seasonality, residual) = self.forward(x)
        self.reconstructions.append(reconstruction)
        self.ground_truth.append(x)
        
    def on_validation_end(self):
        ground_truth, reconstructions = torch.stack(self.reconstructions), torch.stack(self.ground_truth)
        mse = mean_squared_error(ground_truth, reconstructions) 
        mae = mean_absolute_error(ground_truth, reconstructions)
        print(f'MSE: {mse}')
        print(f'MAE: {mae}')