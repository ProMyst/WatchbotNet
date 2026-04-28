"""
Parquet writer — buffers feature data and flushes to Parquet periodically.

Every feature vector the neural net sees gets written to Parquet for future training.
This is the sub-second data lake.
"""

import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timezone
import time
import threading


class ParquetWriter:
    """Buffered Parquet writer for real-time feature data."""

    def __init__(self, output_dir="./data/parquet", flush_interval=300,
                 compression="zstd"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flush_interval = flush_interval
        self.compression = compression

        self._buffer = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._total_rows = 0

    def write(self, row: dict):
        """Add a row to the buffer."""
        row["recorded_at"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._buffer.append(row)

        # Auto-flush if interval elapsed
        if time.time() - self._last_flush >= self.flush_interval:
            self.flush()

    def write_batch(self, rows: list):
        """Add multiple rows."""
        ts = datetime.now(timezone.utc).isoformat()
        for row in rows:
            row["recorded_at"] = ts
        with self._lock:
            self._buffer.extend(rows)

    def flush(self):
        """Write buffer to Parquet file."""
        with self._lock:
            if not self._buffer:
                return
            data = self._buffer.copy()
            self._buffer.clear()

        # Generate filename with timestamp
        now = datetime.now(timezone.utc)
        filename = f"swaps_{now.strftime('%Y%m%d_%H%M%S')}.parquet"
        path = self.output_dir / filename

        try:
            table = pa.Table.from_pylist(data)
            pq.write_table(table, path, compression=self.compression)
            self._total_rows += len(data)
            self._last_flush = time.time()
        except Exception as e:
            # Put data back if write fails
            with self._lock:
                self._buffer = data + self._buffer
            raise

    @property
    def buffered_rows(self):
        return len(self._buffer)

    @property
    def total_rows_written(self):
        return self._total_rows
