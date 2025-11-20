import torch
import torch.nn as nn
import torch.optim as optim

#
source_sentences = [["I", "love", "cats"], ["She", "reads", "books"]]

target_sentences = [
    ["J'aime", "les", "chats", "<EOS>"],
    ["Elle", "lit", "des", "livres"],
]

# Build vocab
src_vocab = {"<PAD>": 0, "I": 1, "love": 2, "cats": 3, "She": 4, "reads": 5, "books": 6}
tgt_vocab = {
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2,
    "J'aime": 3,
    "les": 4,
    "chats": 5,
    "Elle": 6,
    "lit": 7,
    "des": 8,
    "livres": 9,
}

src_idx = [[src_vocab[w] for w in s] for s in source_sentences]
tgt_idx = [[tgt_vocab["<SOS>"]] + [tgt_vocab[w] for w in t] for t in target_sentences]

src_tensor = torch.tensor(src_idx)
tgt_tensor = torch.tensor(tgt_idx)

d_model = 16
seq_len_src = src_tensor.shape[1]
seq_len_tgt = tgt_tensor.shape[1]
vocab_size_src = len(src_vocab)
vocab_size_tgt = len(tgt_vocab)

src_emb = nn.Embedding(vocab_size_src, d_model)
tgt_emb = nn.Embedding(vocab_size_tgt, d_model)


def pos_enc(seq_len, d_model):
    import math

    pe = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
    return pe


src_pe = pos_enc(seq_len_src, d_model)
tgt_pe = pos_enc(seq_len_tgt, d_model)

src_inp = src_emb(src_tensor) + src_pe
tgt_inp = tgt_emb(tgt_tensor) + tgt_pe


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads=2, d_ff=32):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x1 = self.norm1(x)
        attn_out, _ = self.attn(x1, x1, x1)
        x2 = x + attn_out
        x3 = self.norm2(x2)
        ffn_out = self.ffn(x3)
        out = x2 + ffn_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads=2, d_ff=32):
        super().__init__()
        self.masked_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.enc_dec_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out):
        # Masked Self-Attention
        x1 = self.norm1(x)
        S_tgt = x.shape[1]
        causal_mask = torch.tril(torch.ones(S_tgt, S_tgt)).bool()
        attn_out1, _ = self.masked_attn(x1, x1, x1, attn_mask=~causal_mask)
        x2 = x + attn_out1

        # Encoder-Decoder Attention
        x3 = self.norm2(x2)
        attn_out2, _ = self.enc_dec_attn(x3, enc_out, enc_out)
        x4 = x2 + attn_out2

        # FFN
        x5 = self.norm3(x4)
        ffn_out = self.ffn(x5)
        out = x4 + ffn_out
        return out


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_tgt, d_model):
        super().__init__()
        self.src_emb = nn.Embedding(vocab_size_src, d_model)
        self.tgt_emb = nn.Embedding(vocab_size_tgt, d_model)
        self.encoder = EncoderBlock(d_model)
        self.decoder = DecoderBlock(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size_tgt)

    def forward(self, src, tgt):
        src_inp = self.src_emb(src)
        tgt_inp = self.tgt_emb(tgt)
        enc_out = self.encoder(src_inp)
        dec_out = self.decoder(tgt_inp, enc_out)
        out = self.fc_out(dec_out)
        return out


model = MiniTransformer(vocab_size_src, vocab_size_tgt, d_model)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(src_tensor, tgt_tensor[:, :-1])  # teacher forcing
    loss = criterion(output.view(-1, vocab_size_tgt), tgt_tensor[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.eval()
tgt_input = torch.tensor([[tgt_vocab["<SOS>"]] * 2]).T  # start with <SOS>
with torch.no_grad():
    for i in range(4):  # max length
        out = model(src_tensor, tgt_input)
        next_token = out.argmax(-1)[:, -1:]  # pick last token
        tgt_input = torch.cat([tgt_input, next_token], dim=1)

print("Generated sequences indices:", tgt_input)
