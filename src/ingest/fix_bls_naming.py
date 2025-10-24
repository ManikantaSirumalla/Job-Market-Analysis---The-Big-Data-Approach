#!/usr/bin/env python3
"""
Fix BLS file naming to preserve year information
"""

import shutil
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.paths import RAW_DIR
from src.common.logs import get_logger

logger = get_logger("ingest.fix_bls")

def fix_bls_naming():
    """Fix BLS file naming to include year information"""
    source_dir = Path("/Users/manikantasirumalla/Desktop/untitled folder/BLS")
    target_dir = RAW_DIR / "bls"
    
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return False
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing files first
    for file_path in target_dir.glob("*.xlsx"):
        file_path.unlink()
        logger.info(f"Removed {file_path.name}")
    
    # Copy BLS files with proper year naming
    bls_files = list(source_dir.glob("*.xlsx"))
    logger.info(f"Found {len(bls_files)} BLS files")
    
    copied_files = []
    for file_path in bls_files:
        # Extract year from filename (e.g., national_M2024_dl.xlsx -> 2024)
        year = file_path.stem.split('_')[1].replace('M', '')
        target_file = target_dir / f"national_{year}.xlsx"
        
        try:
            shutil.copy2(file_path, target_file)
            copied_files.append(target_file.name)
            logger.info(f"✅ Copied {file_path.name} -> {target_file.name}")
        except Exception as e:
            logger.error(f"❌ Failed to copy {file_path.name}: {e}")
    
    logger.info(f"✅ BLS files renamed successfully. {len(copied_files)} files copied to {target_dir}")
    return True

if __name__ == "__main__":
    fix_bls_naming()
