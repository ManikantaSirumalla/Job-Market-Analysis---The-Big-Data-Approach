
import argparse, requests, sys, pathlib

# Add project root to Python path
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.paths import RAW_DIR
from src.common.logs import get_logger

logger = get_logger("ingest.stackoverflow")

def main(year: int):
    url = f"https://insights.stackoverflow.com/survey/{year}/survey_results_public.csv"
    out_dir = RAW_DIR / "stackoverflow"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"survey_{year}.csv"
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        logger.warning(f"Year {year} not available at {url}")
        return
    fp.write_bytes(r.content)
    logger.info(f"Saved {fp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    args = ap.parse_args()
    main(args.year)
