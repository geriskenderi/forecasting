import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device, bi=False, dropout=0.1):
        # Instantiate variables
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        # Build layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=bi)
        in_fc = hidden_dim * 2 if bi else hidden_dim
        self.fc = nn.Linear(in_fc, output_dim)

    def forward(self, x, future):
        # Init hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # Init cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # LSTM cell outputs and current states
        out, (ht, ct) = self.lstm(x, (h0.detach(), c0.detach()))

        # Take the wanted future steps through the linear layer
        predictions = self.fc(out[:, -future:, :])

        return predictions


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device, bi=False, dropout=0.1):
        # Instantiate variables
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        # Build layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=bi)
        in_fc = hidden_dim * 2 if bi else hidden_dim
        self.fc = nn.Linear(in_fc, output_dim)

    def forward(self, x, future):
        # Init hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # LSTM cell outputs and current states
        out, ht = self.gru(x, h0.detach())

        # Take the wanted future steps through the linear layer
        predictions = self.fc(out[:, -future:, :])

        return predictions
