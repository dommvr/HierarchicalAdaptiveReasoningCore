import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_mult=4, dropout=0.1):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)