
"""Minimal bronze->silver cleaner for GHArchive and BLS (toy version)."""
import json, gzip, pathlib, pandas as pd
from src.common.paths import RAW_DIR, SILVER_DIR
from src.common.logs import get_logger

logger = get_logger("etl.bronze_to_silver")

def normalize_gharchive_day(date: str):
    day_dir = RAW_DIR / "gharchive" / date
    rows = []
    for gz in sorted(day_dir.glob("*.json.gz")):
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    rows.append({
                        "created_at": ev.get("created_at"),
                        "type": ev.get("type"),
                        "repo": (ev.get("repo") or {}).get("name"),
                        "lang": (ev.get("payload") or {}).get("language") or None
                    })
                except Exception:
                    continue
    if not rows:
        logger.warning("No events parsed; skipping.")
        return
    df = pd.DataFrame(rows)
    out = SILVER_DIR / "gharchive"
    out.mkdir(parents=True, exist_ok=True)
    fp = out / f"events_{date}.parquet"
    df.to_parquet(fp, index=False)
    logger.info(f"Wrote {fp} with {len(df)} rows.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()
    normalize_gharchive_day(args.date)
