#!/bin/bash
# WWAI-Macro Platform Startup Script
# Starts all services: Landing Page, GNN API, GNN Frontend

echo "=========================================="
echo "  WWAI-Macro Platform Startup"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base paths
MACRO_BASE="/mnt/nas/WWAI/WWAI-MACRO"
GNN_API_PATH="$MACRO_BASE/WWAI-GNN/api"
GNN_FRONTEND_PATH="$MACRO_BASE/WWAI-GNN/frontend"
LANDING_PATH="$MACRO_BASE/wwai-macro-landing"

# Function to check if port is in use
check_port() {
    lsof -i :$1 > /dev/null 2>&1
    return $?
}

# Start GNN API (Port 8005)
echo -e "${BLUE}Starting GNN API on port 8005...${NC}"
if check_port 8005; then
    echo "  Port 8005 already in use, skipping..."
else
    cd "$GNN_API_PATH"
    python main.py > /tmp/gnn_api.log 2>&1 &
    echo -e "  ${GREEN}GNN API started${NC}"
fi

# Start GNN Frontend (Port 3789)
echo -e "${BLUE}Starting GNN Frontend on port 3789...${NC}"
if check_port 3789; then
    echo "  Port 3789 already in use, skipping..."
else
    cd "$GNN_FRONTEND_PATH"
    npm run dev > /tmp/gnn_frontend.log 2>&1 &
    echo -e "  ${GREEN}GNN Frontend started${NC}"
fi

# Start Landing Page (Port 3801)
echo -e "${BLUE}Starting Landing Page on port 3801...${NC}"
if check_port 3801; then
    echo "  Port 3801 already in use, skipping..."
else
    cd "$LANDING_PATH"
    npm run dev > /tmp/landing.log 2>&1 &
    echo -e "  ${GREEN}Landing Page started${NC}"
fi

# Wait for services to start
sleep 5

echo ""
echo "=========================================="
echo "  Services Status"
echo "=========================================="
echo ""
echo "Landing Page:   http://localhost:3801"
echo "GNN Dashboard:  http://localhost:3789"
echo "GNN API:        http://localhost:8005"
echo ""
echo "VAR Dashboard:  http://localhost:8012 (start separately)"
echo ""
echo "=========================================="
echo "  Log files:"
echo "=========================================="
echo "  GNN API:      /tmp/gnn_api.log"
echo "  GNN Frontend: /tmp/gnn_frontend.log"
echo "  Landing:      /tmp/landing.log"
echo ""
echo "To stop all services: pkill -f 'next dev' && pkill -f 'uvicorn'"
