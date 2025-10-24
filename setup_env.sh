#!/bin/bash

# Job Market Analysis Environment Setup Script

echo "Setting up Job Market Analysis environment..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/raw data/bronze data/silver data/gold configs tests

# Set PYTHONPATH for this session
export PYTHONPATH="/Users/manikantasirumalla/Desktop/job-market-analysis:$PYTHONPATH"

echo "Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "source .venv/bin/activate"
echo "export PYTHONPATH=\"/Users/manikantasirumalla/Desktop/job-market-analysis:\$PYTHONPATH\""
echo ""
echo "To run the API:"
echo "make api"
echo ""
echo "To test ingestion (with PYTHONPATH set):"
echo "python src/ingest/download_gharchive.py --date 2024-01-01"
