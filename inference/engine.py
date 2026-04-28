"""
Live inference engine — scores all watched tokens every second via GPU.

Workflow:
  1. Pool monitor pushes swap events → FeatureBuffer updates
  2. Every second, export buffer → batch GPU inference
  3. Top scoring tokens → trade signal
  4. Every feature vector → Parquet for future training
"""

import torch
import time
import threading
import logging
from typing import Optional, Callable

from data.features import FeatureBuffer
from data.parquet_writer import ParquetWriter
from config import get_config

log = logging.getLogger("inference")


class InferenceEngine:
    """Real-time GPU inference engine."""

    def __init__(self, model, on_signal: Callable = None):
        """
        Args:
            model: trained PyTorch model (PumpLSTM, PumpCNN, or NeuralEnsemble)
            on_signal: callback(pair_address, score, reg_pred) when signal fires
        """
        cfg = get_config()
        self.device = cfg["gpu"]["device"]
        self.batch_size = cfg["gpu"]["batch_inference_size"]

        # Model
        self.model = model.to(self.device)
        self.model.eval()

        # Feature buffer (GPU memory)
        self.buffer = FeatureBuffer(
            feature_dim=cfg["model"]["feature_dim"],
            seq_len=cfg["model"]["sequence_length"],
            max_tokens=cfg["pools"]["max_watched"],
            device=self.device,
        )

        # Parquet writer (records everything for training)
        self.parquet = ParquetWriter(
            output_dir=cfg["data"]["parquet_dir"],
            flush_interval=cfg["data"]["flush_interval_sec"],
        )

        # Trading thresholds
        self.pump_threshold = cfg["trading"].get("pump_threshold", 0.60)
        self.reg_threshold = cfg["trading"].get("reg_threshold", 10.0)

        # Callback
        self.on_signal = on_signal

        # Stats
        self.scores_computed = 0
        self.signals_fired = 0
        self._running = False

    def start(self):
        """Start the inference loop in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        log.info(f"Inference engine started | device={self.device} | "
                 f"max_tokens={self.buffer.max_tokens}")

    def stop(self):
        self._running = False
        self.parquet.flush()

    def on_swap(self, pair_address: str, swap_data: dict):
        """Called by the pool monitor when a swap event arrives.

        Updates the feature buffer for this token.
        """
        # Compute per-swap features
        features = self._compute_swap_features(swap_data)

        # Update buffer
        self.buffer.update(pair_address, features)

        # Record to Parquet
        record = {"pair_address": pair_address, **features, **swap_data}
        self.parquet.write(record)

    def _compute_swap_features(self, swap: dict) -> dict:
        """Transform raw swap event into feature vector."""
        return {
            "price_norm": swap.get("price", 0),
            "price_change_1s": swap.get("price_change", 0),
            "volume_1s": swap.get("amount_usd", 0),
            "buy_volume_1s": swap.get("amount_usd", 0) if swap.get("side") == "buy" else 0,
            "sell_volume_1s": swap.get("amount_usd", 0) if swap.get("side") == "sell" else 0,
            "buy_count_1s": 1 if swap.get("side") == "buy" else 0,
            "sell_count_1s": 1 if swap.get("side") == "sell" else 0,
            "buy_ratio": 1.0 if swap.get("side") == "buy" else 0.0,
            "liquidity": swap.get("liquidity", 0),
            "liq_change": swap.get("liq_change", 0),
            "trade_intensity": 1,
            "large_buy_flag": 1 if swap.get("amount_usd", 0) > 1000 and swap.get("side") == "buy" else 0,
            "large_sell_flag": 1 if swap.get("amount_usd", 0) > 1000 and swap.get("side") == "sell" else 0,
        }

    def _inference_loop(self):
        """Main loop — score all tokens every second."""
        while self._running:
            try:
                start = time.time()

                # Get batch from buffer
                x_seq, x_ctx, pairs = self.buffer.get_batch()

                if x_seq is not None and len(pairs) > 0:
                    # GPU inference
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            with torch.cuda.amp.autocast():
                                probs, reg_pred, attn = self.model.predict(x_seq, x_ctx)
                        else:
                            probs, reg_pred, attn = self.model.predict(x_seq, x_ctx)

                    self.scores_computed += len(pairs)

                    # Check for signals
                    pump_probs = probs[:, 1].cpu().numpy()  # P(pump)
                    reg_preds = reg_pred.squeeze().cpu().numpy()

                    for i, pair in enumerate(pairs):
                        p_pump = float(pump_probs[i])
                        r_pred = float(reg_preds[i]) if reg_preds.ndim > 0 else float(reg_preds)

                        if p_pump >= self.pump_threshold and r_pred >= self.reg_threshold:
                            self.signals_fired += 1
                            if self.on_signal:
                                self.on_signal(pair, p_pump, r_pred)

                elapsed = time.time() - start
                # Sleep remainder of 1 second
                sleep_time = max(0, 1.0 - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                log.error(f"Inference error: {e}", exc_info=True)
                time.sleep(1)

    def get_status(self) -> dict:
        return {
            "active_tokens": self.buffer.active_count,
            "scores_computed": self.scores_computed,
            "signals_fired": self.signals_fired,
            "parquet_buffered": self.parquet.buffered_rows,
            "parquet_total": self.parquet.total_rows_written,
        }
