#!/usr/bin/env python3
"""
Copy existing StackOverflow and BLS data into the project structure
"""

import shutil
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.paths import RAW_DIR
from src.common.logs import get_logger

logger = get_logger("ingest.copy_data")

def copy_stackoverflow_data():
    """Copy StackOverflow survey data from untitled folder to project structure"""
    source_dir = Path("/Users/manikantasirumalla/Desktop/untitled folder/StackOverflow/Datasets")
    target_dir = RAW_DIR / "stackoverflow"
    
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return False
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy survey results
    survey_files = list(source_dir.glob("survey_results_public*.csv"))
    logger.info(f"Found {len(survey_files)} StackOverflow survey files")
    
    copied_files = []
    for file_path in survey_files:
        # Extract year from filename
        year = file_path.stem.split()[-1]
        target_file = target_dir / f"survey_{year}.csv"
        
        try:
            shutil.copy2(file_path, target_file)
            copied_files.append(target_file.name)
            logger.info(f"✅ Copied {file_path.name} -> {target_file.name}")
        except Exception as e:
            logger.error(f"❌ Failed to copy {file_path.name}: {e}")
    
    # Copy schema files
    schema_dir = Path("/Users/manikantasirumalla/Desktop/untitled folder/StackOverflow/Dataset Schemas")
    if schema_dir.exists():
        schema_files = list(schema_dir.glob("survey_results_schema*.csv"))
        logger.info(f"Found {len(schema_files)} schema files")
        
        for file_path in schema_files:
            year = file_path.stem.split()[-1]
            target_file = target_dir / f"schema_{year}.csv"
            
            try:
                shutil.copy2(file_path, target_file)
                copied_files.append(target_file.name)
                logger.info(f"✅ Copied schema {file_path.name} -> {target_file.name}")
            except Exception as e:
                logger.error(f"❌ Failed to copy schema {file_path.name}: {e}")
    
    logger.info(f"✅ StackOverflow data copy complete. {len(copied_files)} files copied to {target_dir}")
    return True

def copy_bls_data():
    """Copy BLS data from untitled folder to project structure"""
    source_dir = Path("/Users/manikantasirumalla/Desktop/untitled folder/BLS")
    target_dir = RAW_DIR / "bls"
    
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return False
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy BLS files
    bls_files = list(source_dir.glob("*.xlsx"))
    logger.info(f"Found {len(bls_files)} BLS files")
    
    copied_files = []
    for file_path in bls_files:
        # Extract year from filename
        year = file_path.stem.split('_')[-1].replace('M', '')
        target_file = target_dir / f"national_{year}.xlsx"
        
        try:
            shutil.copy2(file_path, target_file)
            copied_files.append(target_file.name)
            logger.info(f"✅ Copied {file_path.name} -> {target_file.name}")
        except Exception as e:
            logger.error(f"❌ Failed to copy {file_path.name}: {e}")
    
    logger.info(f"✅ BLS data copy complete. {len(copied_files)} files copied to {target_dir}")
    return True

def get_data_summary():
    """Get summary of all available data"""
    logger.info("📊 Data Summary:")
    
    # StackOverflow data
    stackoverflow_dir = RAW_DIR / "stackoverflow"
    if stackoverflow_dir.exists():
        so_files = list(stackoverflow_dir.glob("*.csv"))
        logger.info(f"  StackOverflow: {len(so_files)} files")
        for file_path in sorted(so_files):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"    - {file_path.name}: {size_mb:.1f} MB")
    
    # BLS data
    bls_dir = RAW_DIR / "bls"
    if bls_dir.exists():
        bls_files = list(bls_dir.glob("*.xlsx"))
        logger.info(f"  BLS: {len(bls_files)} files")
        for file_path in sorted(bls_files):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"    - {file_path.name}: {size_mb:.1f} MB")
    
    # GitHub data
    github_dir = RAW_DIR / "github"
    if github_dir.exists():
        github_files = list(github_dir.glob("*.json.gz"))
        total_size_gb = sum(f.stat().st_size for f in github_files) / (1024**3)
        logger.info(f"  GitHub Archive: {len(github_files)} files, {total_size_gb:.2f} GB")

def main():
    """Main function to copy all existing data"""
    logger.info("🚀 Starting data copy process...")
    
    # Copy StackOverflow data
    logger.info("\n📊 Copying StackOverflow survey data...")
    so_success = copy_stackoverflow_data()
    
    # Copy BLS data
    logger.info("\n📈 Copying BLS data...")
    bls_success = copy_bls_data()
    
    # Show summary
    logger.info("\n📊 Data Summary:")
    get_data_summary()
    
    if so_success and bls_success:
        logger.info("\n✅ All data copied successfully!")
    else:
        logger.warning("\n⚠️  Some data copy operations failed. Check logs above.")

if __name__ == "__main__":
    main()
