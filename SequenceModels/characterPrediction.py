# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

# %%

text = """
From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
But thou contracted to thine own bright eyes,
Feed'st thy light's flame with self-substantial fuel,
Making a famine where abundance lies,
Thy self thy foe, to thy sweet self too cruel.
Thou that art now the world's fresh ornament,
And only herald to the gaudy spring,
Within thine own bud buriest thy content,
And tender churl mak’st waste in niggarding:
Pity the world, or else this glutton be,
To eat the world's due, by the grave and thee.
"""

vocab = sorted(list(set(text)))
vocab_size = len(vocab)
print(f"Length of vocabulary: {vocab_size}")

# %%

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}


# %%
def encode(s: str) -> list[int]:
    """Converts a string to a list of integer indices."""
    return [stoi[ch] for ch in s]


def decode(indices: list[int] | torch.Tensor) -> str:
    """Converts a list/tensor of integer indices back to a string."""
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    return "".join([itos[idx] for idx in indices])


# %%

encoded_text = torch.tensor(encode(text), dtype=torch.long)
SEQ_LEN = 40
BATCH_SIZE = 64


class CharDataset(Dataset):
    def __init__(self, data, seq_len=40):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):

        chunk = self.data[idx : idx + self.seq_len]
        target = self.data[idx + 1 : idx + self.seq_len + 1]
        return chunk, target


dataset = CharDataset(encoded_text, SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Dataset size:", len(dataset))

# %%


class CharPredictions(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm1 = nn.LSTM(embedding_dim, hidden_size, num_layers=3, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):

        out = self.embedding(x)

        out, _ = self.lstm1(out)

        logits = self.fc(out)

        return logits


cp = CharPredictions(vocab_size).to(DEVICE)
optimiser = torch.optim.AdamW(cp.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# %%


def train(epochs, data_loader, model, optimizer, criterion):
    model.train()
    print(f"Starting training for {epochs} epochs on device: {DEVICE}")
    for epoch in range(1, epochs + 1):
        total_loss = 0

        for chunk, target in data_loader:
            chunk, target = chunk.to(DEVICE), target.to(DEVICE)

            logits = model(chunk)
            logits_for_loss = logits.permute(0, 2, 1)
            loss = criterion(logits_for_loss, target)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch}/{epochs}: Average Loss = {avg_loss:.4f}")


train(epochs=100, data_loader=loader, model=cp, optimizer=optimiser, criterion=loss_fn)


# %%


def generate_predictions(model, seed_text, max_length=100):
    model.eval()

    initial_indices = encode(seed_text)
    if len(initial_indices) < SEQ_LEN:
        print("Error: Seed text must be at least SEQ_LEN characters.")
        return ""

    current_chunk = (
        torch.tensor(initial_indices[-SEQ_LEN:], dtype=torch.long)
        .unsqueeze(0)
        .to(DEVICE)
    )

    generated_indices = []

    with torch.no_grad():

        for _ in range(max_length):

            logits = model(current_chunk)

            next_char_logits = logits[:, -1, :]

            next_char_index = torch.argmax(next_char_logits, dim=1)

            generated_indices.append(next_char_index.item())

            next_token_tensor = next_char_index.unsqueeze(1)

            current_chunk = torch.cat([current_chunk[:, 1:], next_token_tensor], dim=1)

    return decode(generated_indices)


seed = text[:SEQ_LEN]
print(f"\n--- Seed Text (SEQ_LEN={SEQ_LEN}) ---\n'{seed}'")


generated_text = generate_predictions(cp, seed, max_length=50)

print(f"\n--- Generated Text (50 chars) ---\n{seed + generated_text}")
