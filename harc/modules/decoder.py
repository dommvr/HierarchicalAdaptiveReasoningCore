import torch
import torch.nn as nn
from harc.modules.transformer_layers import TransformerDecoderLayer
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=4, n_heads=8, max_len=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.to_logits = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tgt_tokens, frozen_slots, tgt_mask=None):
        """
        tgt_tokens: (B, L_t)
        frozen_slots: (B, G, K, 2, d)
        """
        B, L_t = tgt_tokens.shape
        x = self.token_emb(tgt_tokens) + self.pos_emb[:, :L_t, :]
        for layer in self.layers:
            x = layer(x, frozen_slots, tgt_mask=tgt_mask)
        logits = self.to_logits(x)
        return logits