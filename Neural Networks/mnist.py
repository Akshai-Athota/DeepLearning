# %%

import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# %%

train_data = datasets.MNIST(
    "./Datasets", train=True, download=True, transform=transform
)
test_data = datasets.MNIST(
    "./Datasets", train=True, download=False, transform=transform
)

print(train_data)

# %%
train_data_loader = DataLoader(train_data, shuffle=True, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)


# %%
data_iter = iter(train_data_loader)
x, y = next(data_iter)

x.shape, y.shape

# %%
torch.unique(y)
# %%
from matplotlib import pyplot as plt

plt.imshow(x[0].squeeze())


# %%
class classifer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.tensor):
        return self.block(x)


model = classifer()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# %%
train_losses, test_losses, accuracies = [], [], []

for epoch in range(10):
    model.train()
    running_loss = 0
    for X, y in train_data_loader:
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    test_loss = 0
    with torch.no_grad():
        for X_test, y_test in test_data_loader:
            y_pred = model(X_test)
            test_loss += criterion(y_pred, y_test).item()
            preds = torch.argmax(y_pred, dim=1)
            correct += (preds == y_test).sum().item()
            total += y_test.size(0)

    acc = correct / total
    train_losses.append(running_loss / len(train_data_loader))
    test_losses.append(test_loss / len(test_data_loader))
    accuracies.append(acc)

    print(
        f"Epoch [{epoch+1}/10] | Train Loss: {train_losses[-1]:.4f} | Test Acc: {acc:.4f}"
    )

# %%

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("Loss Curves")

plt.subplot(1, 2, 2)
plt.plot(accuracies, label="Test Accuracy", color="orange")
plt.legend()
plt.title("Test Accuracy over Epochs")
plt.show()

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_true, y_pred_all = [], []
model.eval()
with torch.no_grad():
    for X_test, y_test in test_data_loader:
        y_pred = model(X_test)
        preds = torch.argmax(y_pred, dim=1)
        y_true.extend(y_test.numpy())
        y_pred_all.extend(preds.numpy())

cm = confusion_matrix(y_true, y_pred_all)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# %%
examples = iter(test_data_loader)
images, labels = next(examples)
with torch.no_grad():
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i].squeeze(), cmap="gray")
    plt.title(f"T:{labels[i].item()}, P:{preds[i].item()}")
    plt.axis("off")
plt.show()

# %%
