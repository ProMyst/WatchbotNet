#!/usr/bin/env python3
"""
Train the neural net on collected data.

Usage:
  python scripts/train.py                    # train with defaults
  python scripts/train.py --model lstm       # LSTM only
  python scripts/train.py --model cnn        # CNN only
  python scripts/train.py --model ensemble   # LSTM + CNN ensemble
  python scripts/train.py --epochs 200       # more epochs
"""

import argparse
import sys
import time
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import load_config
from models import PumpLSTM, PumpCNN, NeuralEnsemble
from data.dataset import SwapSequenceDataset

cfg = load_config()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=cfg["model"]["type"], choices=["lstm", "cnn", "ensemble"])
    p.add_argument("--epochs", type=int, default=cfg["training"]["epochs"])
    p.add_argument("--batch-size", type=int, default=cfg["training"]["batch_size"])
    p.add_argument("--lr", type=float, default=cfg["training"]["learning_rate"])
    p.add_argument("--data-dir", default="./data/raw")
    p.add_argument("--output-dir", default=cfg["data"]["models_dir"])
    return p.parse_args()


def load_data(data_dir):
    """Load and prepare training data from pulled VPS data."""
    data_dir = Path(data_dir)
    db_path = data_dir / "watchdog.db"

    print(f"Loading data from {data_dir}...")

    db = sqlite3.connect(str(db_path))

    # Signal outcomes (labeled data)
    signals = pd.read_sql("""
        SELECT * FROM signal_outcomes
        WHERE source IN ('solana', 'newpair')
        AND status = 'resolved'
        AND pct_change_1h IS NOT NULL
        ORDER BY signal_timestamp
    """, db)
    signals["signal_timestamp"] = pd.to_datetime(signals["signal_timestamp"])
    signals["pct_change_1h"] = pd.to_numeric(signals["pct_change_1h"], errors="coerce")

    # Newpair snapshots (for sequences)
    snapshots = pd.read_sql("""
        SELECT * FROM newpair_snapshots ORDER BY recorded_at
    """, db)
    snapshots["recorded_at"] = pd.to_datetime(snapshots["recorded_at"])

    db.close()

    # Create labels
    signals["label"] = 0  # flat
    signals.loc[signals["pct_change_1h"] > 3.0, "label"] = 1   # pump
    signals.loc[signals["pct_change_1h"] < -5.0, "label"] = 2  # dump

    # Add derived features
    for col in ["hour_of_day", "day_of_week", "fear_greed_score", "btc_dominance",
                "btc_price_change_24h", "sol_price_change_24h", "liquidity_usd",
                "market_cap", "vol_liq_turnover"]:
        signals[col] = pd.to_numeric(signals[col], errors="coerce")

    import math
    signals["hour_sin"] = signals["hour_of_day"].apply(lambda h: math.sin(2 * math.pi * h / 24) if pd.notna(h) else 0)
    signals["hour_cos"] = signals["hour_of_day"].apply(lambda h: math.cos(2 * math.pi * h / 24) if pd.notna(h) else 0)
    signals["day_sin"] = signals["day_of_week"].apply(lambda d: math.sin(2 * math.pi * d / 7) if pd.notna(d) else 0)
    signals["day_cos"] = signals["day_of_week"].apply(lambda d: math.cos(2 * math.pi * d / 7) if pd.notna(d) else 0)
    signals["is_solana"] = (signals["source"] == "solana").astype(float)
    signals["is_newpair"] = (signals["source"] == "newpair").astype(float)

    liq = pd.to_numeric(signals["liquidity_usd"], errors="coerce").fillna(0)
    mcap = pd.to_numeric(signals["market_cap"], errors="coerce").fillna(1)
    signals["liq_to_mcap"] = liq / mcap.clip(lower=1)

    signals = signals.dropna(subset=["pct_change_1h"])

    print(f"  Signals: {len(signals):,} ({(signals.label==1).sum()} pump / "
          f"{(signals.label==2).sum()} dump / {(signals.label==0).sum()} flat)")
    print(f"  Snapshots: {len(snapshots):,}")

    return signals, snapshots


def create_model(model_type, feature_dim, num_classes=3, dropout=0.2):
    if model_type == "lstm":
        return PumpLSTM(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
    elif model_type == "cnn":
        return PumpCNN(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
    elif model_type == "ensemble":
        return NeuralEnsemble(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(args):
    signals, snapshots = load_data(args.data_dir)

    # Time-based split
    n = len(signals)
    t1 = int(n * cfg["training"]["train_split"])
    t2 = int(n * (cfg["training"]["train_split"] + cfg["training"]["val_split"]))

    train_signals = signals.iloc[:t1]
    val_signals = signals.iloc[t1:t2]
    test_signals = signals.iloc[t2:]

    print(f"  Train: {len(train_signals):,} | Val: {len(val_signals):,} | Test: {len(test_signals):,}")

    # Create datasets
    feature_dim = 32
    train_ds = SwapSequenceDataset(train_signals, snapshots, seq_len=cfg["model"]["sequence_length"])
    val_ds = SwapSequenceDataset(val_signals, snapshots, seq_len=cfg["model"]["sequence_length"])
    test_ds = SwapSequenceDataset(test_signals, snapshots, seq_len=cfg["model"]["sequence_length"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    device = cfg["gpu"]["device"] if torch.cuda.is_available() else "cpu"
    model = create_model(args.model, feature_dim=feature_dim)
    model = model.to(device)
    print(f"\n  Model: {args.model} | Params: {sum(p.numel() for p in model.parameters()):,} | Device: {device}")

    # Class weights (handle imbalance)
    label_counts = np.bincount(train_ds.labels, minlength=3).astype(float)
    class_weights = 1.0 / (label_counts + 1)
    class_weights = class_weights / class_weights.sum() * 3
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # Loss
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights)
    reg_criterion = nn.SmoothL1Loss()

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float("inf")
    patience = cfg["training"]["early_stopping_patience"]
    patience_counter = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Training for {args.epochs} epochs (patience={patience})...")
    print(f"  {'Epoch':<8} {'Train Loss':>12} {'Val Loss':>12} {'Val Acc':>10} {'LR':>12}")
    print(f"  {'-'*56}")

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            x_seq, x_ctx, labels, targets = [b.to(device) for b in batch]

            cls_logits, reg_pred, _ = model(x_seq, x_ctx)
            loss_cls = cls_criterion(cls_logits, labels)
            loss_reg = reg_criterion(reg_pred.squeeze(), targets) * 0.1
            loss = loss_cls + loss_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                x_seq, x_ctx, labels, targets = [b.to(device) for b in batch]
                cls_logits, reg_pred, _ = model(x_seq, x_ctx)
                loss = cls_criterion(cls_logits, labels) + reg_criterion(reg_pred.squeeze(), targets) * 0.1
                val_loss += loss.item()
                preds = cls_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  {epoch:<8} {train_loss:>12.4f} {val_loss:>12.4f} {val_acc:>9.1%} {lr:>12.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_type": args.model,
                "feature_dim": feature_dim,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": cfg,
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
                break

    # Test
    print(f"\n  Loading best model (val_loss={best_val_loss:.4f})...")
    checkpoint = torch.load(output_dir / "best_model.pt")
    model.load_state_dict(checkpoint["model_state"])

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            x_seq, x_ctx, labels, targets = [b.to(device) for b in batch]
            cls_logits, _, _ = model(x_seq, x_ctx)
            probs = torch.softmax(cls_logits, dim=1)
            preds = cls_logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = (all_preds == all_labels).mean()

    print(f"\n  TEST RESULTS:")
    print(f"    Accuracy: {test_acc:.1%}")
    for cls, name in [(0, "flat"), (1, "pump"), (2, "dump")]:
        mask = all_labels == cls
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == cls).mean()
            print(f"    {name}: {cls_acc:.1%} ({mask.sum()} samples)")

    print(f"\n  Model saved to {output_dir / 'best_model.pt'}")
    print(f"  Done!")


if __name__ == "__main__":
    args = parse_args()
    train(args)
