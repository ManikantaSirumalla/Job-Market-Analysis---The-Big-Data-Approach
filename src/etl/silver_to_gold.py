
"""Toy silver->gold aggregation: count events by repo and type."""
import pandas as pd
from pathlib import Path
from src.common.paths import SILVER_DIR, GOLD_DIR
from src.common.logs import get_logger

logger = get_logger("etl.silver_to_gold")

def aggregate_events(date: str):
    src = SILVER_DIR / "gharchive" / f"events_{date}.parquet"
    if not src.exists():
        logger.error(f"Missing {src}. Run bronze_to_silver first.")
        return
    df = pd.read_parquet(src)
    agg = df.groupby(["repo","type"], dropna=False).size().reset_index(name="count")
    out_dir = GOLD_DIR / "gharchive"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"events_summary_{date}.parquet"
    agg.to_parquet(out_fp, index=False)
    logger.info(f"Wrote {out_fp} with {len(agg)} rows.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()
    aggregate_events(args.date)
