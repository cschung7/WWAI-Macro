#!/bin/bash
# WWAI-Macro Setup Script for Server 97 (Backup)
# Run this script on server 163.239.155.97 to set up the backup environment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

INSTALL_DIR="/mnt/nas/WWAI/WWAI-MACRO"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  WWAI-Macro Backup Server Setup${NC}"
echo -e "${BLUE}  Server: 163.239.155.97${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running on correct server
CURRENT_IP=$(hostname -I | awk '{print $1}')
echo -e "Current server IP: ${YELLOW}$CURRENT_IP${NC}"

# Create directory if needed
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo ""
echo -e "${YELLOW}Step 1: Cloning repositories...${NC}"

# Clone Landing Page
if [ ! -d "wwai-macro-landing" ]; then
    echo "Cloning WWAI-Macro (Landing Page)..."
    git clone git@github.com:cschung7/WWAI-Macro.git wwai-macro-landing
else
    echo "wwai-macro-landing exists, pulling latest..."
    cd wwai-macro-landing && git pull && cd ..
fi

# Clone GNN
if [ ! -d "WWAI-GNN" ]; then
    echo "Cloning WWAI-GNN..."
    git clone git@github.com:cschung7/WWAI-GNN.git
else
    echo "WWAI-GNN exists, pulling latest..."
    cd WWAI-GNN && git pull && cd ..
fi

# Clone GraphECast
if [ ! -d "WWAI-GraphECast" ]; then
    echo "Cloning WWAI-GraphECast..."
    git clone git@github.com:cschung7/WWAI-GraphECast.git
else
    echo "WWAI-GraphECast exists, pulling latest..."
    cd WWAI-GraphECast && git pull && cd ..
fi

# Clone Macro-Risk (VAR)
if [ ! -d "WWAI-VAR" ]; then
    echo "Cloning Macro-Risk (VAR)..."
    git clone git@github.com:cschung7/Macro-Risk.git WWAI-VAR
else
    echo "WWAI-VAR exists, pulling latest..."
    cd WWAI-VAR && git pull && cd ..
fi

echo ""
echo -e "${YELLOW}Step 2: Installing dependencies...${NC}"

# Landing Page
echo "Installing Landing Page dependencies..."
cd "$INSTALL_DIR/wwai-macro-landing"
npm install

# GNN Frontend
echo "Installing GNN Frontend dependencies..."
cd "$INSTALL_DIR/WWAI-GNN/frontend"
npm install

# Update environment files for server 97
echo ""
echo -e "${YELLOW}Step 3: Configuring environment for Server 97...${NC}"

# Landing page .env.local
cat > "$INSTALL_DIR/wwai-macro-landing/.env.local" << 'EOF'
# Server 97 Configuration
NEXT_PUBLIC_SERVER_IP=163.239.155.97
EOF

# GNN Frontend .env.local
cat > "$INSTALL_DIR/WWAI-GNN/frontend/.env.local" << 'EOF'
# GNN API URL for Server 97
NEXT_PUBLIC_API_URL=http://163.239.155.97:8005
EOF

# Update landing page to use server 97 IP
echo "Updating landing page server IP..."
sed -i 's/163.239.155.96/163.239.155.97/g' "$INSTALL_DIR/wwai-macro-landing/app/page.tsx"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Start all services with:"
echo "   cd $INSTALL_DIR && ./start_all_services.sh"
echo ""
echo "Or start individually:"
echo ""
echo "   # Landing Page (port 3801)"
echo "   cd $INSTALL_DIR/wwai-macro-landing && npm run dev"
echo ""
echo "   # GNN Dashboard (port 3789)"
echo "   cd $INSTALL_DIR/WWAI-GNN/frontend && npm run dev"
echo ""
echo "   # GNN API (port 8005)"
echo "   cd $INSTALL_DIR/WWAI-GNN/api && uvicorn main:app --host 0.0.0.0 --port 8005"
echo ""
echo "   # VAR Dashboard (port 8012)"
echo "   cd $INSTALL_DIR/WWAI-VAR/api && uvicorn main:app --host 0.0.0.0 --port 8012"
echo ""
