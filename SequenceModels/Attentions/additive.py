import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attn_dim):
        super().__init__()
        self.W1 = nn.Linear(encoder_dim, attn_dim)
        self.W2 = nn.Linear(decoder_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, encoder_out, decoder_hidden):

        dec_hidden_expanded = decoder_hidden.unsqueeze(1)

        score = self.v(
            torch.tanh(self.W1(encoder_out) + self.W2(dec_hidden_expanded))
        ).squeeze(-1)

        attention_weights = torch.softmax(score, dim=1)

        context = torch.sum(attention_weights.unsqueeze(-1) * encoder_out, dim=1)

        return context, attention_weights
