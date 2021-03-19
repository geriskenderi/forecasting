import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set up CUDA


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bi=False, dropout=0.1):
        # Instantiate variables
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Build layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bi)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, future):
        # Init hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)

        # Init cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)

        # LSTM cell outputs and current states
        out, (ht, ct) = self.lstm(x, (h0.detach(), c0.detach()))

        # Take the wanted future steps through the linear layer
        predictions = self.linear(out[:, -future:, :])

        return predictions
