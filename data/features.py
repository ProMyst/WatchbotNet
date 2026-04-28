"""
Real-time feature buffer for live inference.

Maintains a rolling window of features per token in GPU memory.
Updated every time a swap event arrives from the WebSocket.
"""

import torch
import numpy as np
from collections import defaultdict
import time


class FeatureBuffer:
    """Rolling feature buffer for live GPU inference.

    Holds the last N seconds of features for each watched token.
    When a swap arrives, updates the buffer. On scoring,
    exports the buffer as a batch tensor for GPU inference.
    """

    def __init__(self, feature_dim=32, seq_len=60, max_tokens=500, device="cuda:0"):
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.device = device

        # Token tracking
        self.token_ids: dict = {}        # pair_address → index
        self.token_info: dict = {}       # pair_address → {mint, symbol, liquidity, ...}
        self._next_id = 0

        # Pre-allocate GPU buffer: (max_tokens, seq_len, feature_dim)
        self.buffer = torch.zeros(
            max_tokens, seq_len, feature_dim,
            device=device, dtype=torch.float32
        )

        # Context features (non-sequential)
        self.context_dim = 16
        self.context = torch.zeros(
            max_tokens, self.context_dim,
            device=device, dtype=torch.float32
        )

        # Timestamps for each token's latest update
        self.last_update = np.zeros(max_tokens)

        # Feature names for sequence (per timestep)
        self.seq_feature_names = [
            "price_norm", "price_change_1s", "volume_1s", "buy_volume_1s",
            "sell_volume_1s", "buy_count_1s", "sell_count_1s", "buy_ratio",
            "liquidity", "liq_change", "spread", "depth_imbalance",
            "price_from_high", "price_from_low", "volatility_10s",
            "volume_acceleration", "buyer_growth_rate", "momentum",
            "rsi_fast", "vwap_deviation", "trade_intensity",
            "large_buy_flag", "large_sell_flag", "consecutive_buys",
            "sol_price_norm", "btc_ret_1m", "sol_ret_1m", "eth_ret_1m",
            "market_breadth", "funding_rate", "fear_greed_norm",
            "time_of_day_sin",
        ]

        # Context feature names
        self.context_feature_names = [
            "fear_greed", "btc_dominance", "btc_24h_change", "sol_24h_change",
            "total_mcap", "btc_funding", "sol_funding", "btc_oi",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "is_new_token", "token_age_min", "liquidity_usd", "mcap_usd",
        ]

    def register_token(self, pair_address: str, info: dict = None):
        """Start tracking a new token."""
        if pair_address in self.token_ids:
            return self.token_ids[pair_address]

        if self._next_id >= self.max_tokens:
            # Evict least recently updated token
            oldest = np.argmin(self.last_update[:self._next_id])
            old_pair = None
            for pa, idx in self.token_ids.items():
                if idx == oldest:
                    old_pair = pa
                    break
            if old_pair:
                del self.token_ids[old_pair]
                del self.token_info[old_pair]
            idx = oldest
        else:
            idx = self._next_id
            self._next_id += 1

        self.token_ids[pair_address] = idx
        self.token_info[pair_address] = info or {}
        self.buffer[idx].zero_()
        self.context[idx].zero_()
        self.last_update[idx] = time.time()
        return idx

    def update(self, pair_address: str, features: dict):
        """Push a new timestep of features for a token.

        Called every time a swap event arrives from the WebSocket.
        Shifts the sequence left by 1 and adds the new features at the end.
        """
        if pair_address not in self.token_ids:
            return

        idx = self.token_ids[pair_address]

        # Shift left (drop oldest, add new at end)
        self.buffer[idx, :-1] = self.buffer[idx, 1:].clone()

        # Fill new timestep
        for i, name in enumerate(self.seq_feature_names):
            val = features.get(name, 0.0)
            self.buffer[idx, -1, i] = float(val) if val is not None else 0.0

        self.last_update[idx] = time.time()

    def update_context(self, pair_address: str, context: dict):
        """Update static context features for a token."""
        if pair_address not in self.token_ids:
            return
        idx = self.token_ids[pair_address]
        for i, name in enumerate(self.context_feature_names):
            val = context.get(name, 0.0)
            self.context[idx, i] = float(val) if val is not None else 0.0

    def get_batch(self, pair_addresses: list = None):
        """Export current buffer as batch tensors for inference.

        Args:
            pair_addresses: specific tokens to score (None = all active)

        Returns:
            x_seq: (N, seq_len, feature_dim) — sequence tensor
            x_ctx: (N, context_dim) — context tensor
            pairs: list of pair_addresses in order
        """
        if pair_addresses is None:
            pairs = list(self.token_ids.keys())
        else:
            pairs = [pa for pa in pair_addresses if pa in self.token_ids]

        if not pairs:
            return None, None, []

        indices = [self.token_ids[pa] for pa in pairs]
        idx_tensor = torch.tensor(indices, device=self.device)

        x_seq = self.buffer[idx_tensor]     # (N, seq_len, feature_dim)
        x_ctx = self.context[idx_tensor]     # (N, context_dim)

        return x_seq, x_ctx, pairs

    def remove_token(self, pair_address: str):
        """Stop tracking a token."""
        if pair_address in self.token_ids:
            idx = self.token_ids.pop(pair_address)
            self.token_info.pop(pair_address, None)
            self.buffer[idx].zero_()
            self.context[idx].zero_()
            self.last_update[idx] = 0

    @property
    def active_count(self):
        return len(self.token_ids)
