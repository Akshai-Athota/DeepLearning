import torch
import torch.nn as nn


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_out, decoder_hidden):

        dec = decoder_hidden.unsqueeze(1)
        score = torch.sum(encoder_out * dec, dim=2)
        attn_weights = torch.softmax(score, dim=1)
        context = torch.sum(encoder_out * attn_weights.unsqueeze(-1), dim=1)

        return context, attn_weights
