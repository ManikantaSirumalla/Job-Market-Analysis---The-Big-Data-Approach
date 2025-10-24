
import argparse, gzip, json, os, pathlib, requests, sys
from datetime import datetime

# Add project root to Python path
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.paths import RAW_DIR
from src.common.logs import get_logger

logger = get_logger("ingest.gharchive")

def fetch_hour(date_str: str, hour: int) -> bytes:
    url = f"http://data.gharchive.org/{date_str}-{hour:02d}.json.gz"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def main(date: str):
    # store as raw/gharchive/YYYY-MM-DD/
    out_dir = RAW_DIR / "gharchive" / date
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for h in range(0, 2):  # keep tiny for first run; bump to 24 later
        try:
            blob = fetch_hour(date, h)
            fp = out_dir / f"{date}-{h:02d}.json.gz"
            fp.write_bytes(blob)
            saved += 1
            logger.info(f"Saved {fp}")
        except Exception as e:
            logger.warning(f"Hour {h:02d} failed: {e}")
    logger.info(f"Done. Saved {saved} files to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()
    # validate date
    datetime.strptime(args.date, "%Y-%m-%d")
    main(args.date)
