#!/bin/bash
# ML Service Health Check Script
# Monitors service health and logs

SERVICE_HOST="${ML_HOST:-0.0.0.0}"
SERVICE_PORT="${ML_PORT:-8001}"
LOG_FILE="logs/ml-service.log"

echo "ML Service Health Check"
echo "========================"
echo "Time: $(date)"
echo ""

# Check if service is running
echo "1. Checking if service is accessible..."
if timeout 5 curl -s http://127.0.0.1:${SERVICE_PORT}/health > /dev/null 2>&1; then
    echo "✓ Service is running and responding"
    
    # Get health status
    echo ""
    echo "2. Getting detailed health status..."
    curl -s http://127.0.0.1:${SERVICE_PORT}/health | python3 -m json.tool 2>/dev/null || \
    curl -s http://127.0.0.1:${SERVICE_PORT}/health
    
    echo ""
    echo "3. Service endpoints:"
    echo "   - Health: http://127.0.0.1:${SERVICE_PORT}/health"
    echo "   - Docs: http://127.0.0.1:${SERVICE_PORT}/docs"
    echo "   - RedDocs: http://127.0.0.1:${SERVICE_PORT}/redoc"
    
else
    echo "✗ Service is NOT responding"
    echo "  Trying to connect to http://127.0.0.1:${SERVICE_PORT}"
    echo ""
    echo "Check logs:"
    if [ -f "$LOG_FILE" ]; then
        echo "  Last 20 lines of $LOG_FILE:"
        tail -20 "$LOG_FILE"
    else
        echo "  No log file found at $LOG_FILE"
    fi
fi

echo ""
echo "4. Recent log entries (if available):"
if [ -f "$LOG_FILE" ]; then
    echo "   Last 10 ERROR entries:"
    grep -i "error\|exception\|failed" "$LOG_FILE" | tail -10 || echo "   No errors found"
    echo ""
    echo "   Last 10 INFO entries:"
    grep -i "info" "$LOG_FILE" | tail -10 || echo "   No info entries"
fi

echo ""
echo "========================"
