# backend/model/cnn_mlp_model.py
import torch
from torch import nn

class LightCurveCNN(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 7, padding=3),    nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(0.1), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),    nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2),    nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.1), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),   nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.AdaptiveAvgPool1d(1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResidualFF(nn.Module):
    def __init__(self, dim, dropout=0.2, expand=2):
        super().__init__()
        hidden = dim * expand
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return x + self.net(x)


class TabularMLP(nn.Module):
    def __init__(self, in_dim, out_dim=32, width=64, depth=2, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(in_dim, width), nn.LayerNorm(width), nn.SiLU()
        )
        self.blocks = nn.Sequential(*[
            ResidualFF(width, dropout=dropout, expand=2) for _ in range(depth)
        ])
        self.outp = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, out_dim), 
            nn.SiLU(), nn.Dropout(dropout)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, t):
        if t.ndim == 1:
            t = t.unsqueeze(0)
        h = self.inp(t)
        h = self.blocks(h)
        return self.outp(h)


class CNNPlusMLPReg(nn.Module):
    def __init__(self, in_ch_lc=3, in_dim_tab=10):
        super().__init__()
        self.cnn = LightCurveCNN(in_ch=in_ch_lc)
        self.tab = TabularMLP(in_dim=in_dim_tab)
        self.head = nn.Sequential(
            nn.Linear(128+32, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, x_lc, x_tab):
        h = torch.cat([self.cnn(x_lc), self.tab(x_tab)], dim=1)
        return self.head(h).squeeze(-1)