#!/bin/bash

echo "🚀 Starting Smart Career Planner..."

# Kill any existing processes
echo "Cleaning up existing processes..."
pkill -f "streamlit\|smart_career_api" 2>/dev/null || true
sleep 2

# Start Smart Career API
echo "Starting Smart Career API on port 8001..."
source .venv/bin/activate
python smart_career_api.py &
API_PID=$!

# Wait for API to start
sleep 3

# Test API
echo "Testing API..."
if curl -s http://127.0.0.1:8001/health > /dev/null; then
    echo "✅ API is running successfully!"
else
    echo "❌ API failed to start"
    exit 1
fi

# Start Optimized Streamlit App
echo "Starting Optimized Streamlit App on port 8503..."
python -m streamlit run optimized_streamlit.py --server.port 8503 --server.headless true &
STREAMLIT_PID=$!

# Wait for Streamlit to start
sleep 5

# Test Streamlit
echo "Testing Streamlit..."
if curl -s -I http://127.0.0.1:8503 | grep -q "200 OK"; then
    echo "✅ Streamlit is running successfully!"
else
    echo "❌ Streamlit failed to start"
    exit 1
fi

echo ""
echo "🎉 Smart Career Planner is now running!"
echo ""
echo "📱 Complete Streamlit Dashboard: http://127.0.0.1:8503"
echo "🔧 Smart Career API: http://127.0.0.1:8001"
echo ""
echo "📊 Available Tabs:"
echo "   • Career Planner - Personalized skill recommendations"
echo "   • Market Overview - Job market KPIs and trends"
echo "   • Skill Analysis - In-demand skills and categories"
echo "   • Salary Intelligence - Salary calculator and trends"
echo "   • Real-time Trends - Live job market insights"
echo "   • About - Project information"
echo ""
echo "Press Ctrl+C to stop all services"

# Keep script running and handle cleanup
trap 'echo "Stopping services..."; kill $API_PID $STREAMLIT_PID 2>/dev/null; exit' INT
wait
