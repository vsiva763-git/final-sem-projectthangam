#!/bin/bash

echo "============================================================"
echo "🚀 Starting Speech Enhancement Web Application"
echo "============================================================"
echo ""

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask..."
    pip install flask werkzeug -q
fi

# Check if model exists
if [ ! -f "demo_checkpoints/checkpoint_epoch_1.pt" ]; then
    echo "🤖 Model not found. Running examples to create it..."
    python examples.py
fi

# Start the web application
echo "🌐 Starting web server..."
echo ""
python app.py
