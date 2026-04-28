"""
Neural ensemble — combines LSTM + CNN predictions.
Can also include the v7 tree model as a third voter.
"""

import torch
import torch.nn as nn
from .lstm_model import PumpLSTM
from .cnn_model import PumpCNN


class NeuralEnsemble(nn.Module):
    def __init__(self, feature_dim=32, num_classes=3, dropout=0.2):
        super().__init__()

        self.lstm = PumpLSTM(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
        self.cnn = PumpCNN(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)

        # Learnable ensemble weights
        self.ensemble_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x_seq, x_context=None):
        lstm_cls, lstm_reg, lstm_attn = self.lstm(x_seq, x_context)
        cnn_cls, cnn_reg, _ = self.cnn(x_seq, x_context)

        # Weighted average
        w = torch.softmax(self.ensemble_weight, dim=0)
        cls_logits = w[0] * lstm_cls + w[1] * cnn_cls
        reg_pred = w[0] * lstm_reg + w[1] * cnn_reg

        return cls_logits, reg_pred, lstm_attn

    def predict(self, x_seq, x_context=None):
        self.eval()
        with torch.no_grad():
            cls_logits, reg_pred, attn = self.forward(x_seq, x_context)
            probs = torch.softmax(cls_logits, dim=1)
        return probs, reg_pred, attn
