#!/usr/bin/env python3
"""
Live trading with the neural net.

Connects to Solana via WebSocket, runs GPU inference,
executes Jupiter swaps, and logs everything to Parquet.

Usage:
  python scripts/live_trade.py                    # live trading
  python scripts/live_trade.py --paper            # paper trading (no real swaps)
  python scripts/live_trade.py --pools 20         # watch 20 pools
"""

import argparse
import sys
import time
import logging
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import load_config
from models import PumpLSTM, PumpCNN, NeuralEnsemble
from inference.engine import InferenceEngine
from data.features import FeatureBuffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("live_trade")

cfg = load_config()


def load_model(model_path="./models/saved/best_model.pt"):
    checkpoint = torch.load(model_path, map_location="cpu")
    model_type = checkpoint.get("model_type", "lstm")
    feature_dim = checkpoint.get("feature_dim", 32)

    if model_type == "lstm":
        model = PumpLSTM(feature_dim=feature_dim)
    elif model_type == "cnn":
        model = PumpCNN(feature_dim=feature_dim)
    elif model_type == "ensemble":
        model = NeuralEnsemble(feature_dim=feature_dim)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    model.load_state_dict(checkpoint["model_state"])
    log.info(f"Loaded {model_type} model from {model_path} "
             f"(epoch {checkpoint.get('epoch', '?')}, val_acc={checkpoint.get('val_acc', '?'):.1%})")
    return model


def on_trade_signal(pair_address, pump_prob, reg_pred):
    """Called when the neural net fires a trade signal."""
    log.info(f"[!] SIGNAL: {pair_address[:16]}... | P(pump)={pump_prob:.3f} | "
             f"predicted return={reg_pred:.1f}%")
    # TODO: integrate Jupiter swap execution from jupiter_live_bot.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true", help="Paper trade only")
    parser.add_argument("--pools", type=int, default=cfg["pools"]["max_watched"])
    parser.add_argument("--model", default="./models/saved/best_model.pt")
    args = parser.parse_args()

    mode = "PAPER" if args.paper else "LIVE"
    log.info(f"Starting WatchbotNet [{mode}] | pools={args.pools}")

    # Load model
    model = load_model(args.model)

    # Create inference engine
    engine = InferenceEngine(model=model, on_signal=on_trade_signal)

    # TODO: Start Helius/RPC WebSocket pool monitor
    # TODO: Connect pool monitor events to engine.on_swap()
    # TODO: Integrate Jupiter swap execution for live mode

    engine.start()

    log.info("Neural net inference engine running. Press Ctrl+C to stop.")
    try:
        while True:
            status = engine.get_status()
            log.info(f"Tokens={status['active_tokens']} | "
                     f"Scored={status['scores_computed']} | "
                     f"Signals={status['signals_fired']} | "
                     f"Parquet={status['parquet_total']} rows")
            time.sleep(15)
    except KeyboardInterrupt:
        log.info("Shutting down...")
        engine.stop()


if __name__ == "__main__":
    main()
