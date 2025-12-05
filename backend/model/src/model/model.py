import torch
import torch.nn as nn


class KeystrokeIDModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(KeystrokeIDModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        # batch_first=True ensures input is (Batch, Seq, Feature) rather than (Seq, Batch, Feature)
        # bidirectional=True lets the model see future context (useful for typing rhythm)
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            bidirectional=True)

        # Fully Connected Layer for Classification
        # We multiply hidden_size * 2 because it is Bi-Directional
        self.fc = nn.Linear(hidden_size * 2, num_classes)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (Batch_Size, Seq_Length, Input_Size)

        # Initialize hidden state and cell state (optional, PyTorch defaults to zeros)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        # out shape: (Batch_Size, Seq_Length, Hidden_Size * 2)
        out, _ = self.lstm(x, (h0, c0))

        # We generally only care about the output of the *last* time step for classification
        # out[:, -1, :] selects the last vector in the sequence
        out = out[:, -1, :]

        out = self.dropout(out)
        out = self.fc(out)
        return out
