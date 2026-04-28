"""
1D CNN model for memecoin pump prediction.

Treats the swap sequence as a 1D signal and applies convolutional filters
to detect patterns (volume spikes, price acceleration, buy pressure shifts).

Faster than LSTM, good at detecting local patterns.
"""

import torch
import torch.nn as nn


class PumpCNN(nn.Module):
    def __init__(self, feature_dim=32, num_classes=3, dropout=0.2):
        super().__init__()

        self.feature_dim = feature_dim
        self.context_dim = 16

        # Conv blocks — detect patterns at different timescales
        self.conv1 = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Multi-scale pooling — capture both recent and longer patterns
        self.pool_recent = nn.AdaptiveAvgPool1d(1)   # global
        self.pool_max = nn.AdaptiveMaxPool1d(1)       # peak signal

        # Context projection
        self.context_proj = nn.Sequential(
            nn.Linear(self.context_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Heads
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self.regressor = nn.Sequential(
            nn.Linear(128 * 2 + 64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq, x_context=None):
        """
        Args:
            x_seq: (batch, seq_len, feature_dim)
            x_context: (batch, context_dim)
        """
        # Conv expects (batch, channels, seq_len)
        x = x_seq.transpose(1, 2)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Pool
        avg = self.pool_recent(x).squeeze(-1)  # (batch, 128)
        mx = self.pool_max(x).squeeze(-1)      # (batch, 128)

        if x_context is not None:
            ctx = self.context_proj(x_context)
        else:
            ctx = torch.zeros(x_seq.size(0), 64, device=x_seq.device)

        combined = torch.cat([avg, mx, ctx], dim=1)

        cls_logits = self.classifier(combined)
        reg_pred = self.regressor(combined)

        return cls_logits, reg_pred, None

    def predict(self, x_seq, x_context=None):
        self.eval()
        with torch.no_grad():
            cls_logits, reg_pred, _ = self.forward(x_seq, x_context)
            probs = torch.softmax(cls_logits, dim=1)
        return probs, reg_pred, None
