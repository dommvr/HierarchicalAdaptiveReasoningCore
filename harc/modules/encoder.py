import torch
import torch.nn as nn
import torch.nn.functional as F
from harc.modules import FeedForward
from harc.modules.transformer_layers import TransformerEncoderLayer


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=4, n_heads=8, max_len=512, G=3, K=4, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        self.group_queries = nn.Parameter(torch.randn(G, d_model))
        self.slot_queries = nn.Parameter(torch.randn(G, K, d_model))

        self.group_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.slot_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)

        self.to_role_context = nn.Linear(d_model, 2 * d_model)

    def forward(self, tokens, mask=None):
        """
        tokens: (B, L)
        returns slots: (B, G, K, 2d)
        """
        B, L = tokens.shape
        x = self.token_emb(tokens) + self.pos_emb[:, :L, :]
        for layer in self.layers:
            x = layer(x, mask)

        # ---- group-level cross-attention ----
        group_q = self.group_queries.unsqueeze(0).expand(B, -1, -1)     # (B, G, d)
        group_sum, _ = self.group_attn(group_q, x, x)                   # (B, G, d)

        # ---- slot=level extraction ----
        slots = []
        for g in range(group_sum.size(1)):
            slot_q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)
            kv = torch.cat([x, group_sum[:, g:g+1, :]], dim=1)
            slot_out, _ = self.slot_attn(slot_q, kv, kv)
            rc = self.to_role_context(slot_out).view(B, -1, 2, slot_out.size(1))
            slots.append(rc)
        slots = torch.stack(slots, dim=1)       # (B, G, K, 2, d)
        return slots