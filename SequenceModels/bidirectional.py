import torch
from torch import nn


class BiLstm(nn.Module):
    def __init__(self, input_size, embeeding_size, num_layers, output_size):
        super().__init__()
        self.emb = nn.Embedding(input_size, embeeding_size)
        self.lstm = nn.LSTM(embeeding_size, 64, 5, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        return self.fc(out[:, -1, :])


class BiGru(nn.Module):
    def __init__(self, input_size, embeeding_size, num_layers, output_size):
        super().__init__()
        self.emb = nn.Embedding(input_size, embeeding_size)
        self.lstm = nn.GRU(embeeding_size, 64, 5, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        return self.fc(out[:, -1, :])
