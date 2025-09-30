import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512, num_blocks=4, dropout_rate=0.3):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Create residual blocks
        self.blocks = nn.ModuleList([
            self._make_block(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])

        # Output layers
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self._init_weights()

    def _make_block(self, dim, dropout_rate):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_bn(x)
        x = F.relu(self.input_layer(x))

        # Apply residual blocks
        for block in self.blocks:
            identity = x
            out = block(x)
            x = F.relu(out + 0.5*identity)  # Residual connection + activation

        x = self.final_layer(x)
        return x