# %%
import torch
from torch import nn


class mlp(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.tensor):
        return self.layers(x)


from sklearn.datasets import make_moons

X, y = make_moons(n_samples=2000, shuffle=True, noise=0.2)

x_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y.reshape((-1, 1)), dtype=torch.float)

model = mlp(2, 5, 1)
criterion = torch.nn.BCELoss()
optimiser = torch.optim.Adam(model.parameters())

for _ in range(500):
    y_preds = model(x_tensor)
    loss = criterion(y_preds, y_tensor)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

print(f"loss : {loss.item()}")

# %%
