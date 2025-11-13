import torch
import torch.nn as nn
import torch.optim as optim


seq = "abcabcabcabcabc"
chars = sorted(list(set(seq)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


input_seq = [char_to_idx[ch] for ch in seq[:-1]]
target_seq = [char_to_idx[ch] for ch in seq[1:]]


x = torch.tensor(input_seq).unsqueeze(1)
y = torch.tensor(target_seq).unsqueeze(1)

vocab_size = len(chars)
hidden_size = 8  # hidden layer size


class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = torch.tanh(self.i2h(combined))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


model = VanillaRNN(vocab_size, hidden_size, vocab_size)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


epochs = 300
for epoch in range(epochs):
    total_loss = 0
    hidden = model.init_hidden()

    for t in range(len(x)):
        input_vec = torch.zeros(1, vocab_size)
        input_vec[0][x[t]] = 1.0

        target = y[t]

        output, hidden = model(input_vec, hidden)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}]  Loss: {total_loss:.4f}")


def predict(start_char="a", length=10):
    model.eval()
    hidden = model.init_hidden()
    input_vec = torch.zeros(1, vocab_size)
    input_vec[0][char_to_idx[start_char]] = 1.0

    result = [start_char]
    for _ in range(length):
        output, hidden = model(input_vec, hidden)
        topv, topi = output.topk(1)
        char_idx = topi[0][0].item()
        result.append(idx_to_char[char_idx])

        input_vec = torch.zeros(1, vocab_size)
        input_vec[0][char_idx] = 1.0

    return "".join(result)


print("\nGenerated sequence:", predict("a", 20))
