# %%
import numpy as np
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(100, 124),
    nn.ReLU(),
    nn.Linear(124, 256),
    nn.ReLU(),
    nn.Linear(256, 124),
    nn.ReLU(),
    nn.Linear(124, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

criterion = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters())

x = torch.randn(64, 100)
y = torch.randint(0, 10, (64,))

for i in range(100):
    model.train()
    y_preds = model(x)
    loss = criterion(y_preds, y)
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimiser.step()

    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item()
    print(
        f"Epoch {i+1} | Loss: {loss.item():.4f} | Grad Norm (after clip): {grad_norm:.4f}"
    )

# %%
