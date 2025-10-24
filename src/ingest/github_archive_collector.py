#!/usr/bin/env python3
"""
GitHub Archive Data Collector
Collects GitHub Archive data (8GB) with proper error handling and validation
"""

import argparse
import gzip
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.paths import RAW_DIR
from src.common.logs import get_logger

logger = get_logger("ingest.github_archive")


class GitHubArchiveCollector:
    """Collect GitHub Archive data (8GB)"""

    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else RAW_DIR / "github"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = 'http://data.gharchive.org'

    def collect_month_data(self, year=2024, month=1):
        """
        Collect one month of GitHub archive data
        Approximately 8GB total
        """
        logger.info(f"🔍 Collecting GitHub Archive data for {year}-{month:02d}")

        # Calculate days in month
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)

        days_in_month = (next_month - datetime(year, month, 1)).days

        downloaded_files = []

        for day in range(1, days_in_month + 1):
            for hour in range(24):
                filename = f"{year}-{month:02d}-{day:02d}-{hour}.json.gz"
                url = f"{self.base_url}/{filename}"
                output_path = self.output_dir / filename

                # Skip if already downloaded
                if output_path.exists():
                    logger.info(f"⏭️  Skipping {filename} (already exists)")
                    continue

                try:
                    logger.info(f"⬇️  Downloading {filename}...")
                    response = requests.get(url, stream=True, timeout=60)

                    if response.status_code == 200:
                        total_size = int(response.headers.get('content-length', 0))

                        with open(output_path, 'wb') as f:
                            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                                    pbar.update(len(chunk))

                        downloaded_files.append(filename)
                        logger.info(f"✅ Downloaded {filename}")
                    else:
                        logger.warning(f"❌ Failed to download {filename}: Status {response.status_code}")

                    # Be nice to the server
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"❌ Error downloading {filename}: {str(e)}")
                    continue

        logger.info(f"✅ Downloaded {len(downloaded_files)} files for {year}-{month:02d}")
        return downloaded_files

    def collect_day_data(self, year=2024, month=1, day=1):
        """
        Collect one day of GitHub archive data
        Useful for testing and smaller data collection
        """
        logger.info(f"🔍 Collecting GitHub Archive data for {year}-{month:02d}-{day:02d}")

        downloaded_files = []

        for hour in range(24):
            filename = f"{year}-{month:02d}-{day:02d}-{hour}.json.gz"
            url = f"{self.base_url}/{filename}"
            output_path = self.output_dir / filename

            # Skip if already downloaded
            if output_path.exists():
                logger.info(f"⏭️  Skipping {filename} (already exists)")
                continue

            try:
                logger.info(f"⬇️  Downloading {filename}...")
                response = requests.get(url, stream=True, timeout=60)

                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))

                    with open(output_path, 'wb') as f:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                                pbar.update(len(chunk))

                    downloaded_files.append(filename)
                    logger.info(f"✅ Downloaded {filename}")
                else:
                    logger.warning(f"❌ Failed to download {filename}: Status {response.status_code}")

                # Be nice to the server
                time.sleep(1)

            except Exception as e:
                logger.error(f"❌ Error downloading {filename}: {str(e)}")
                continue

        logger.info(f"✅ Downloaded {len(downloaded_files)} files for {year}-{month:02d}-{day:02d}")
        return downloaded_files

    def validate_data(self, sample_size=10):
        """Validate downloaded GitHub data"""
        logger.info("🔍 Validating GitHub Archive data...")

        files = list(self.output_dir.glob("*.json.gz"))
        if not files:
            logger.warning("No files found to validate")
            return 0, []

        valid_files = 0
        invalid_files = []

        # Sample validation to avoid processing all files
        sample_files = files[:sample_size] if len(files) > sample_size else files

        for file_path in tqdm(sample_files, desc="Validating files"):
            try:
                with gzip.open(file_path, 'rt') as f:
                    # Try to read first few lines
                    for i, line in enumerate(f):
                        if i >= 5:  # Check first 5 lines
                            break
                        json.loads(line)
                valid_files += 1
            except Exception as e:
                invalid_files.append((file_path.name, str(e)))

        logger.info(f"✅ Valid files: {valid_files}")
        if invalid_files:
            logger.warning(f"❌ Invalid files: {len(invalid_files)}")
            for filename, error in invalid_files:
                logger.warning(f"   - {filename}: {error}")

        return valid_files, invalid_files

    def get_file_stats(self):
        """Get statistics about downloaded files"""
        files = list(self.output_dir.glob("*.json.gz"))
        total_size = sum(f.stat().st_size for f in files)
        
        logger.info(f"📊 File Statistics:")
        logger.info(f"   Total files: {len(files)}")
        logger.info(f"   Total size: {total_size / (1024**3):.2f} GB")
        
        return len(files), total_size


def main():
    parser = argparse.ArgumentParser(description="GitHub Archive Data Collector")
    parser.add_argument("--year", type=int, default=2024, help="Year to collect data for")
    parser.add_argument("--month", type=int, default=1, help="Month to collect data for")
    parser.add_argument("--day", type=int, help="Specific day to collect data for (if not provided, collects entire month)")
    parser.add_argument("--validate", action="store_true", help="Validate downloaded data")
    parser.add_argument("--stats", action="store_true", help="Show file statistics")
    parser.add_argument("--output-dir", help="Output directory (default: data/raw/github)")

    args = parser.parse_args()

    collector = GitHubArchiveCollector(args.output_dir)

    if args.stats:
        collector.get_file_stats()
        return

    if args.validate:
        collector.validate_data()
        return

    if args.day:
        # Collect specific day
        collector.collect_day_data(args.year, args.month, args.day)
    else:
        # Collect entire month
        collector.collect_month_data(args.year, args.month)

    # Show stats after collection
    collector.get_file_stats()


if __name__ == "__main__":
    main()
