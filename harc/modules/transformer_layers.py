import torch.nn as nn
from harc.modules import FeedForward

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.ff = FeedForward(dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x
    

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout)
        self.ff = FeedForward(dim, dropout=dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, memory, tgt_mask=None, mean_mask=None):
        self_attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self_attn_out)

        cross_attn_out, _ = self.cross_attn(x, memory, memory, key_padding_mask=mean_mask)
        x = self.norm2(x + cross_attn_out)
        x = self.norm3(x + self.ff(x))
        return x