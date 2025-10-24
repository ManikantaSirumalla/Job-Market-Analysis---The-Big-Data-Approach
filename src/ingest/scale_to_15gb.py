#!/usr/bin/env python3
"""
Scale to 15+ GB Data Collection Strategy
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.paths import RAW_DIR
from src.common.logs import get_logger

logger = get_logger("ingest.scale")

def calculate_collection_plan():
    """Calculate how much data we need to collect to reach 15+ GB"""
    
    # Current data size
    current_size = 0
    for file_path in RAW_DIR.rglob("*"):
        if file_path.is_file():
            current_size += file_path.stat().st_size
    
    current_gb = current_size / (1024**3)
    target_gb = 15.0
    needed_gb = target_gb - current_gb
    
    logger.info(f"📊 Current data size: {current_gb:.2f} GB")
    logger.info(f"🎯 Target size: {target_gb:.2f} GB")
    logger.info(f"📈 Need to collect: {needed_gb:.2f} GB")
    
    # GitHub Archive data estimation
    # Each hour file is typically 50-150 MB
    # Each day = 24 hours = ~1-3 GB
    # Each month = ~30-90 GB
    
    avg_hour_size_mb = 100  # Conservative estimate
    hours_per_day = 24
    days_per_month = 30
    
    gb_per_day = (avg_hour_size_mb * hours_per_day) / 1024
    gb_per_month = gb_per_day * days_per_month
    
    days_needed = needed_gb / gb_per_day
    months_needed = needed_gb / gb_per_month
    
    logger.info(f"\n📅 Collection Strategy:")
    logger.info(f"   Average file size: {avg_hour_size_mb} MB per hour")
    logger.info(f"   GB per day: {gb_per_day:.2f} GB")
    logger.info(f"   GB per month: {gb_per_month:.2f} GB")
    logger.info(f"   Days needed: {days_needed:.1f} days")
    logger.info(f"   Months needed: {months_needed:.1f} months")
    
    return {
        'current_gb': current_gb,
        'needed_gb': needed_gb,
        'days_needed': days_needed,
        'months_needed': months_needed
    }

def suggest_collection_strategies():
    """Suggest different collection strategies"""
    
    logger.info(f"\n🚀 Collection Strategies to Reach 15+ GB:")
    
    logger.info(f"\n1️⃣  RECENT DATA STRATEGY (Recommended)")
    logger.info(f"   Collect last 6-8 weeks of data")
    logger.info(f"   Command: python src/ingest/massive_github_collector.py --mode recent-weeks --weeks 8")
    logger.info(f"   Expected: ~8-12 GB additional data")
    logger.info(f"   Time: 2-4 hours")
    
    logger.info(f"\n2️⃣  MULTI-MONTH STRATEGY")
    logger.info(f"   Collect 2-3 months of data")
    logger.info(f"   Command: python src/ingest/massive_github_collector.py --mode month-range --start-year 2024 --start-month 1 --end-year 2024 --end-month 3")
    logger.info(f"   Expected: ~10-15 GB additional data")
    logger.info(f"   Time: 4-8 hours")
    
    logger.info(f"\n3️⃣  HIGH-ACTIVITY STRATEGY")
    logger.info(f"   Collect high-activity days from multiple months")
    logger.info(f"   Command: python src/ingest/massive_github_collector.py --mode high-activity --start-year 2024 --start-month 1")
    logger.info(f"   Expected: ~5-8 GB additional data")
    logger.info(f"   Time: 1-2 hours")
    
    logger.info(f"\n4️⃣  MASSIVE COLLECTION STRATEGY")
    logger.info(f"   Collect 6+ months of data")
    logger.info(f"   Command: python src/ingest/massive_github_collector.py --mode month-range --start-year 2023 --start-month 7 --end-year 2024 --end-month 12")
    logger.info(f"   Expected: ~20-30 GB additional data")
    logger.info(f"   Time: 8-16 hours")

def main():
    """Main function"""
    logger.info("🚀 Scaling to 15+ GB Data Collection Plan")
    logger.info("=" * 50)
    
    # Calculate current status
    plan = calculate_collection_plan()
    
    # Suggest strategies
    suggest_collection_strategies()
    
    logger.info(f"\n💡 RECOMMENDATION:")
    if plan['needed_gb'] < 5:
        logger.info(f"   Start with RECENT DATA STRATEGY (8 weeks)")
    elif plan['needed_gb'] < 10:
        logger.info(f"   Use MULTI-MONTH STRATEGY (2-3 months)")
    else:
        logger.info(f"   Use MASSIVE COLLECTION STRATEGY (6+ months)")
    
    logger.info(f"\n⚠️  IMPORTANT NOTES:")
    logger.info(f"   - GitHub Archive has rate limits - be patient")
    logger.info(f"   - Each hour file is 50-150 MB")
    logger.info(f"   - Use --validate flag to check data integrity")
    logger.info(f"   - Monitor disk space during collection")
    logger.info(f"   - Consider running overnight for large collections")

if __name__ == "__main__":
    main()
