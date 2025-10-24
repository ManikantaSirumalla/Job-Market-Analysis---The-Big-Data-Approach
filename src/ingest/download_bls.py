
import argparse, requests, sys, pathlib

# Add project root to Python path
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.paths import RAW_DIR
from src.common.logs import get_logger

logger = get_logger("ingest.bls")

def main(series: str):
    # mirrors BLS time.series folder structure: e.g., 'oe' Occupational Employment
    base = "https://download.bls.gov/pub/time.series"
    url = f"{base}/{series}/{series}.series"
    out_dir = RAW_DIR / "bls" / series
    out_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    fp = out_dir / f"{series}.series"
    fp.write_bytes(r.content)
    logger.info(f"Saved {fp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="oe", help="BLS series code (e.g., oe, ce, la)")
    args = ap.parse_args()
    main(args.series)
