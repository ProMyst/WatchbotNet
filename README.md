# WatchbotNet

Neural network trading system for Solana memecoins. Watches the entire Solana DEX market in real-time, scores tokens via GPU batch inference, and executes trades through Jupiter.

## Architecture

```
Solana RPC/Helius WebSocket → Raw swap stream
  ↓
Feature Buffer (GPU memory, rolling 60s per token)
  ↓
LSTM/CNN Batch Inference (2080 Super, all tokens in 2ms)
  ↓
Top scores → Jupiter Swap API → Real SOL trades
  ↓
Parquet Data Lake → Continuous retraining
```

## Directory Structure

```
WatchbotNet/
├── config/           # Model hyperparams, trading params, API keys
├── data/             # Data loading, Parquet I/O, feature engineering
├── models/           # Neural net architectures (LSTM, CNN, Transformer)
├── training/         # Training loops, walk-forward validation
├── inference/        # Live inference engine, GPU batch scoring
├── pipeline/         # Helius WebSocket, Raydium pool monitor, swap stream
├── trading/          # Jupiter swap execution, position management, exits
├── utils/            # Logging, metrics, visualization
├── scripts/          # CLI tools: train, backtest, deploy, monitor
└── notebooks/        # Jupyter notebooks for research/analysis
```

## Hardware Requirements

- **GPU:** NVIDIA 2080 Super (8GB VRAM) minimum
- **RAM:** 16GB+
- **Storage:** SSD for model + Parquet data
- **OS:** Windows 10/11 with CUDA 11.8+

## Setup (Windows)

```bash
# Clone
git clone https://github.com/ProMyst/WatchbotNet.git
cd WatchbotNet

# Create conda environment
conda create -n watchbot python=3.11
conda activate watchbot

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Copy config
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your API keys

# Pull training data from VPS
python scripts/pull_data.py

# Train
python scripts/train.py

# Paper trade
python scripts/paper_trade.py

# Deploy live
python scripts/live_trade.py
```

## Data Sources

- **VPS Watchdog DB:** 62K+ labeled signal outcomes, 170K+ token snapshots
- **Live Parquet:** Sub-second swap data recorded during inference
- **Rejection outcomes:** What tokens did after we skipped them
- **Trade history:** Every trade with full features, PnL, tx signatures
