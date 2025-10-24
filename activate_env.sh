#!/bin/bash
# Job Market Analysis Environment Activation Script

# Activate virtual environment
source .venv/bin/activate

# Set PYTHONPATH to include project root
export PYTHONPATH="/Users/manikantasirumalla/Desktop/job-market-analysis:$PYTHONPATH"

echo "Environment activated!"
echo "PYTHONPATH set to: $PYTHONPATH"
echo ""
echo "Available commands:"
echo "  make api          - Start FastAPI server"
echo "  make test         - Run tests"
echo "  make format       - Format code with black"
echo ""
echo "Ingestion scripts:"
echo "  python src/ingest/download_gharchive.py --date YYYY-MM-DD"
echo "  python src/ingest/download_bls.py --series oe"
echo "  python src/ingest/download_stackoverflow.py"
