# %%
import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


model = SimpleCNN()
print(model)

x = torch.randn(1, 1, 28, 28)
out = model(x)
print("Output shape:", out.shape)

# %%
import matplotlib.pyplot as plt

plt.imshow(x[0, 0].detach())

# %%
plt.imshow(out.detach())
# %%
