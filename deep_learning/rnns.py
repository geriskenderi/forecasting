import torch
import torch.nn as nn
import pytorch_lightning as pl


class RNN(pl.LightiningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device, bi=False, dropout=0.1, rnn_type='LSTM'):
        # Instantiate variables
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.bidirectional = bi

        # Build layers
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=bi)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=bi)

        in_fc = hidden_dim * 2 if bi else hidden_dim
        self.fc = nn.Linear(in_fc, output_dim)

    def forward(self, x, future):
        # Run through RNN
        out, _ = self.rnn(x)

        # Take the wanted future steps through the linear layer
        predictions = self.fc(out[:, -future:, :])

        return predictions
