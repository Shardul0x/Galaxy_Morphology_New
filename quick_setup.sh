#!/bin/bash
# quick_setup.sh - Run this to set everything up

echo "🚀 Quick Setup - Astrophysics Data Augmentation"
echo "================================================"

# Create directories
mkdir -p config results/web checkpoints

# Install streamlit if needed
pip install streamlit plotly pyyaml -q

echo "✓ Directories created"
echo "✓ Dependencies installed"
echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the web interface:"
echo "  streamlit run web_app.py"
echo ""
echo "To run command line:"
echo "  python run_pipeline.py GalaxyZoo1_DR_table2.csv galaxy_morphology 3"
