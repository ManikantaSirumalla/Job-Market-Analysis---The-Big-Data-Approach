
from pathlib import Path
import os

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

for p in (RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR):
    p.mkdir(parents=True, exist_ok=True)
