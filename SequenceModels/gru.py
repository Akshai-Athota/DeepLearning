# %%
import torch
from torch import nn


class SimpleGru(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, out_features=output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


model = SimpleGru(10, 64, 5, 3)
x = torch.randn(32, 5, 10)
y = model(x)
print(y.shape)
# %%
