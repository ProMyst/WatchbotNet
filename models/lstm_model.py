"""
LSTM model for memecoin pump prediction.

Sees a sequence of swap data (last 60 seconds) and predicts:
  - P(pump): probability of 10%+ gain in next 30 min
  - P(dump): probability of 5%+ loss in next 30 min
  - P(flat): probability of sideways movement

Input: (batch_size, sequence_length, feature_dim)
  - sequence_length = 60 (one sample per second)
  - feature_dim = 32 (price, volume, buy/sell, liquidity, etc.)

Output: (batch_size, 3) — softmax probabilities [pump, dump, flat]
"""

import torch
import torch.nn as nn


class PumpLSTM(nn.Module):
    def __init__(self, feature_dim=32, hidden_size=128, num_layers=2,
                 dropout=0.2, num_classes=3):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input normalization
        self.input_norm = nn.LayerNorm(feature_dim)

        # LSTM backbone — processes the time series
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False,  # causal — can't look into the future
        )

        # Attention layer — learn which timesteps matter most
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Context features (non-sequential: BTC price, fear/greed, etc.)
        self.context_dim = 16
        self.context_proj = nn.Sequential(
            nn.Linear(self.context_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

        # Regression head — predicted return percentage
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x_seq, x_context=None):
        """
        Args:
            x_seq: (batch, seq_len, feature_dim) — time series of swap data
            x_context: (batch, context_dim) — static context features (optional)

        Returns:
            cls_logits: (batch, num_classes) — classification logits
            reg_pred: (batch, 1) — predicted return %
            attention_weights: (batch, seq_len) — which timesteps mattered
        """
        batch_size = x_seq.size(0)

        # Normalize input
        x_seq = self.input_norm(x_seq)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x_seq)
        # lstm_out: (batch, seq_len, hidden_size)

        # Attention — weight each timestep
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1)    # (batch, seq_len)

        # Weighted sum of LSTM outputs
        context_vec = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_out                     # (batch, seq_len, hidden)
        ).squeeze(1)  # (batch, hidden_size)

        # Add static context features if provided
        if x_context is not None:
            ctx = self.context_proj(x_context)  # (batch, hidden//2)
            combined = torch.cat([context_vec, ctx], dim=1)
        else:
            # Zero-pad context
            ctx = torch.zeros(batch_size, self.hidden_size // 2, device=x_seq.device)
            combined = torch.cat([context_vec, ctx], dim=1)

        # Classify and regress
        cls_logits = self.classifier(combined)
        reg_pred = self.regressor(combined)

        return cls_logits, reg_pred, attn_weights

    def predict(self, x_seq, x_context=None):
        """Inference — returns probabilities and predicted return."""
        self.eval()
        with torch.no_grad():
            cls_logits, reg_pred, attn = self.forward(x_seq, x_context)
            probs = torch.softmax(cls_logits, dim=1)
        return probs, reg_pred, attn
