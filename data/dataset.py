"""
PyTorch dataset for swap sequence data.

Each sample is a (sequence, context, label) tuple:
  - sequence: (seq_len, feature_dim) — 60 seconds of per-second features
  - context: (context_dim,) — static context (BTC price, fear/greed, etc.)
  - label: int — 0=flat, 1=pump, 2=dump
  - target: float — actual pct_change in next 30 min
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SwapSequenceDataset(Dataset):
    """Dataset built from signal_outcomes + newpair_snapshots."""

    def __init__(self, signals_df, snapshots_df=None, seq_len=60, label_col="label",
                 target_col="pct_change_1h", context_cols=None):
        """
        Args:
            signals_df: DataFrame with signal features + outcomes
            snapshots_df: DataFrame with newpair price curve snapshots
            seq_len: number of timesteps per sequence
            label_col: column with class labels (0=flat, 1=pump, 2=dump)
            target_col: column with regression target
            context_cols: list of static context feature columns
        """
        self.seq_len = seq_len
        self.context_cols = context_cols or [
            "fear_greed_score", "btc_dominance", "btc_price_change_24h",
            "sol_price_change_24h", "total_market_cap_usd",
            "btc_funding_rate", "sol_funding_rate", "btc_open_interest",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "is_solana", "is_newpair", "liq_to_mcap", "vol_liq_turnover",
        ]

        # Sequence features (per-timestep)
        self.seq_features = [
            "price_usd", "price_change_pct", "volume_5m", "volume_accel",
            "liquidity_usd", "liquidity_change_pct",
            "txns_buys", "txns_sells", "buy_sell_ratio",
            "buyer_count", "buyer_growth",
            "market_cap", "fdv",
            "momentum_score", "bp_score", "volume_surge_ratio",
            "price_from_high", "price_volatility",
            "sol_price", "btc_price", "eth_price",
            "sol_ret_1m", "btc_ret_1m",
            "spread_pct", "depth_ratio",
            "time_since_creation", "snapshot_count",
            "green_candle", "volume_per_liq",
            "buy_pressure", "sell_pressure",
            "net_flow",
        ]

        self.signals = signals_df
        self.snapshots = snapshots_df
        self.labels = signals_df[label_col].values.astype(np.int64)
        self.targets = signals_df[target_col].values.astype(np.float32)

        # Pre-build sequences from snapshots
        self._build_sequences()

    def _build_sequences(self):
        """Build sequence tensors from snapshot data."""
        self.sequences = []
        self.contexts = []

        for idx, row in self.signals.iterrows():
            pair = row.get("pair_address", "")
            ts = row.get("signal_timestamp")

            # Build sequence from snapshots
            seq = np.zeros((self.seq_len, len(self.seq_features)), dtype=np.float32)

            if self.snapshots is not None and pair:
                mask = (self.snapshots["pair_address"] == pair)
                if hasattr(ts, "isoformat"):
                    mask = mask & (self.snapshots["recorded_at"] <= ts)
                recent = self.snapshots[mask].tail(self.seq_len)

                for i, (_, snap) in enumerate(recent.iterrows()):
                    offset = self.seq_len - len(recent) + i
                    for j, feat in enumerate(self.seq_features):
                        if feat in snap.index:
                            val = snap[feat]
                            if pd.notna(val):
                                try:
                                    seq[offset, j] = float(val)
                                except (ValueError, TypeError):
                                    pass

            self.sequences.append(seq)

            # Context features
            ctx = np.zeros(len(self.context_cols), dtype=np.float32)
            for j, col in enumerate(self.context_cols):
                if col in row.index:
                    val = row[col]
                    if pd.notna(val):
                        try:
                            ctx[j] = float(val)
                        except (ValueError, TypeError):
                            pass
            self.contexts.append(ctx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        ctx = torch.tensor(self.contexts[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return seq, ctx, label, target
