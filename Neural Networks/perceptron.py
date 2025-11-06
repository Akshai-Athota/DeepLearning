# %%
import numpy as np
from activationfunctions import Activations

np.random.seed(42)


class Perceptron:
    def __init__(self, epochs: float, lr: float, activation, der_activation):
        self.epochs = epochs
        self.lr = lr
        self.activation = activation
        self.der_activation = der_activation
        self.w = None
        self.b = 0.0

    def forward(self, x):
        n, m = x.shape
        if m != self.w.shape[0]:
            z = np.dot(x, self.w.T) + self.b
        else:
            z = np.dot(x, self.w) + self.b

        return self.activation(z), z

    def fit(self, X, y):
        n, m = X.shape
        self.w = np.random.normal(0, 1, m)
        self.w = np.reshape(self.w, shape=(-1, 1))
        y = y.reshape((-1, 1))
        losses = []
        for _ in range(self.epochs):
            y_hat, z = self.forward(X)
            error = y_hat - y
            dw = np.dot(X.T, error * self.activation(z)) / y.shape[0]
            db = np.mean(error * self.der_activation(z))
            self.w -= self.lr * dw
            self.b -= self.lr * db
            loss = np.mean((y - y_hat) ** 2)
            losses.append(loss)
        return losses

    def predict(self, X):
        y_hat, _ = self.forward(X)
        return (y_hat > 0.5).astype(int)


from sklearn.datasets import make_classification
from matplotlib import pyplot as plt

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=1,
)

activation = Activations()
model = Perceptron(
    activation=activation.relu, der_activation=activation.der_relu, lr=0.1, epochs=500
)
losses = model.fit(X, y)

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

# %%
import torch
from torch import nn


class PytorchPrecepton(nn.Module):
    def __init__(self, n_inputs: int):
        super().__init__()
        self.n_inputs = n_inputs
        self.activation = torch.nn.ReLU()
        self.layer = torch.nn.Linear(n_inputs, 1)

    def forward(self, x: np.ndarray):
        return self.activation(self.layer(x))


pp = PytorchPrecepton(n_inputs=2)
criterion = torch.nn.BCEWithLogitsLoss()
optim = torch.optim.SGD(pp.parameters(), lr=0.1)


# %%
for _ in range(500):
    z = torch.tensor(X, dtype=torch.float)
    y_preds = pp(z)
    y_actual = torch.tensor(y.reshape(-1, 1), dtype=torch.float)
    loss = criterion(y_preds, y_actual)
    optim.zero_grad()
    loss.backward()
    optim.step()

print(f"loss : {loss.item()}")
# %%
