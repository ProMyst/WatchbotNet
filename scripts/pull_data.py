#!/usr/bin/env python3
"""Pull training data from VPS to local machine."""

import os
import subprocess
from pathlib import Path
from config import load_config

cfg = load_config()
host = cfg["vps_host"]
user = cfg["vps_user"]

DATA_DIR = Path("./data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

files = [
    (f"{cfg['vps_watchdog_db']}", DATA_DIR / "watchdog.db"),
    (f"{cfg['vps_trade_logs']}jupiter_paper_trades.csv", DATA_DIR / "jupiter_paper_trades.csv"),
    (f"{cfg['vps_trade_logs']}jupiter_paper_rejections.csv", DATA_DIR / "jupiter_paper_rejections.csv"),
    (f"{cfg['vps_trade_logs']}jupiter_paper_outcomes.csv", DATA_DIR / "jupiter_paper_outcomes.csv"),
    (f"{cfg['vps_trade_logs']}jupiter_live_trades.csv", DATA_DIR / "jupiter_live_trades.csv"),
    (f"{cfg['vps_trade_logs']}jupiter_live_tax.csv", DATA_DIR / "jupiter_live_tax.csv"),
    (f"{cfg['vps_trade_logs']}jupiter_live_rejections.csv", DATA_DIR / "jupiter_live_rejections.csv"),
    (f"{cfg['vps_trade_logs']}jupiter_live_outcomes.csv", DATA_DIR / "jupiter_live_outcomes.csv"),
]

print("Pulling data from VPS...")
for remote, local in files:
    print(f"  {remote} → {local}")
    subprocess.run(["scp", f"{user}@{host}:{remote}", str(local)], timeout=120)

print(f"\nDone. Data saved to {DATA_DIR}")
print(f"Files: {len(list(DATA_DIR.iterdir()))}")
