#!/usr/bin/env python3
"""
Test script to verify the job market analysis environment is working correctly.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import pandas as pd
        print(f"✓ pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import fastapi
        print(f"✓ fastapi {fastapi.__version__}")
    except ImportError as e:
        print(f"✗ fastapi import failed: {e}")
        return False
    
    try:
        import uvicorn
        print(f"✓ uvicorn {uvicorn.__version__}")
    except ImportError as e:
        print(f"✗ uvicorn import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    try:
        import spacy
        print(f"✓ spacy {spacy.__version__}")
    except ImportError as e:
        print(f"✗ spacy import failed: {e}")
        return False
    
    try:
        import nltk
        print(f"✓ nltk {nltk.__version__}")
    except ImportError as e:
        print(f"✗ nltk import failed: {e}")
        return False
    
    try:
        import streamlit
        print(f"✓ streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"✗ streamlit import failed: {e}")
        return False
    
    try:
        import plotly
        print(f"✓ plotly {plotly.__version__}")
    except ImportError as e:
        print(f"✗ plotly import failed: {e}")
        return False
    
    return True

def test_project_structure():
    """Test that the project structure is correct."""
    print("\nTesting project structure...")
    
    required_dirs = [
        "src",
        "src/api",
        "src/common", 
        "src/ingest",
        "src/etl",
        "src/ml",
        "data",
        "data/raw",
        "data/bronze", 
        "data/silver",
        "data/gold",
        "notebooks",
        "dags"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} missing")
            return False
    
    return True

def test_project_imports():
    """Test that project modules can be imported."""
    print("\nTesting project module imports...")
    
    # Add project root to Python path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    try:
        from src.common.paths import RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR
        print("✓ src.common.paths")
    except ImportError as e:
        print(f"✗ src.common.paths import failed: {e}")
        return False
    
    try:
        from src.common.logs import get_logger
        print("✓ src.common.logs")
    except ImportError as e:
        print(f"✗ src.common.logs import failed: {e}")
        return False
    
    try:
        from src.api.app import app
        print("✓ src.api.app")
    except ImportError as e:
        print(f"✗ src.api.app import failed: {e}")
        return False
    
    return True

def test_data_directories():
    """Test that data directories are writable."""
    print("\nTesting data directory permissions...")
    
    data_dirs = ["data/raw", "data/bronze", "data/silver", "data/gold"]
    
    for dir_path in data_dirs:
        path = Path(dir_path)
        if path.exists() and os.access(path, os.W_OK):
            print(f"✓ {dir_path} writable")
        else:
            print(f"✗ {dir_path} not writable")
            return False
    
    return True

def main():
    """Run all tests."""
    print("Job Market Analysis Environment Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_project_structure, 
        test_project_imports,
        test_data_directories
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! Environment is ready.")
        print("\nNext steps:")
        print("1. Run 'make api' to start the FastAPI server")
        print("2. Visit http://127.0.0.1:8000/docs for API documentation")
        print("3. Run ingestion scripts with proper PYTHONPATH")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
