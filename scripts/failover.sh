#!/bin/bash
# WWAI-Macro Failover Script
# Switches landing page between Server 96 (primary) and Server 97 (backup)

set -e

LANDING_DIR="/mnt/nas/WWAI/WWAI-MACRO/wwai-macro-landing"
PAGE_FILE="$LANDING_DIR/app/page.tsx"

PRIMARY_SERVER="163.239.155.96"
BACKUP_SERVER="163.239.155.97"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

show_status() {
    echo -e "${YELLOW}=== Current Server Configuration ===${NC}"
    CURRENT=$(grep "const SERVER_IP" "$PAGE_FILE" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+" || echo "Not found")

    if [[ "$CURRENT" == "$PRIMARY_SERVER" ]]; then
        echo -e "Active Server: ${GREEN}$PRIMARY_SERVER (Primary)${NC}"
    elif [[ "$CURRENT" == "$BACKUP_SERVER" ]]; then
        echo -e "Active Server: ${YELLOW}$BACKUP_SERVER (Backup)${NC}"
    else
        echo -e "Active Server: ${RED}Unknown ($CURRENT)${NC}"
    fi
}

switch_to_primary() {
    echo -e "${GREEN}Switching to PRIMARY server ($PRIMARY_SERVER)...${NC}"
    sed -i "s/$BACKUP_SERVER/$PRIMARY_SERVER/g" "$PAGE_FILE"
    echo "Done. Restart the landing page service to apply changes."
}

switch_to_backup() {
    echo -e "${YELLOW}Switching to BACKUP server ($BACKUP_SERVER)...${NC}"
    sed -i "s/$PRIMARY_SERVER/$BACKUP_SERVER/g" "$PAGE_FILE"
    echo "Done. Restart the landing page service to apply changes."
}

health_check() {
    echo -e "${YELLOW}=== Health Check ===${NC}"

    echo -n "Server 96 (Primary): "
    if curl -s --connect-timeout 3 "http://$PRIMARY_SERVER:3801" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Online${NC}"
    else
        echo -e "${RED}✗ Offline${NC}"
    fi

    echo -n "Server 97 (Backup):  "
    if curl -s --connect-timeout 3 "http://$BACKUP_SERVER:3801" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Online${NC}"
    else
        echo -e "${RED}✗ Offline${NC}"
    fi

    echo ""
    echo "Service Status on Primary ($PRIMARY_SERVER):"
    for port in 3801 8012 3789 8005; do
        echo -n "  Port $port: "
        if curl -s --connect-timeout 2 "http://$PRIMARY_SERVER:$port" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${RED}✗${NC}"
        fi
    done

    echo ""
    echo "Service Status on Backup ($BACKUP_SERVER):"
    for port in 3801 8012 3789 8005; do
        echo -n "  Port $port: "
        if curl -s --connect-timeout 2 "http://$BACKUP_SERVER:$port" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${RED}✗${NC}"
        fi
    done
}

auto_failover() {
    echo -e "${YELLOW}=== Auto Failover Check ===${NC}"

    PRIMARY_OK=false
    BACKUP_OK=false

    if curl -s --connect-timeout 3 "http://$PRIMARY_SERVER:3801" > /dev/null 2>&1; then
        PRIMARY_OK=true
    fi

    if curl -s --connect-timeout 3 "http://$BACKUP_SERVER:3801" > /dev/null 2>&1; then
        BACKUP_OK=true
    fi

    CURRENT=$(grep "const SERVER_IP" "$PAGE_FILE" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+" || echo "")

    if [[ "$PRIMARY_OK" == true ]] && [[ "$CURRENT" != "$PRIMARY_SERVER" ]]; then
        echo "Primary is back online. Switching back..."
        switch_to_primary
    elif [[ "$PRIMARY_OK" == false ]] && [[ "$BACKUP_OK" == true ]] && [[ "$CURRENT" != "$BACKUP_SERVER" ]]; then
        echo -e "${RED}Primary is DOWN! Failing over to backup...${NC}"
        switch_to_backup
    elif [[ "$PRIMARY_OK" == true ]]; then
        echo -e "${GREEN}Primary server is healthy. No action needed.${NC}"
    else
        echo -e "${RED}WARNING: Both servers appear to be offline!${NC}"
    fi
}

restart_services() {
    echo -e "${YELLOW}Restarting landing page...${NC}"
    cd "$LANDING_DIR"

    # Kill existing process on port 3801
    fuser -k 3801/tcp 2>/dev/null || true
    sleep 2

    # Start in background
    nohup npm run dev > /tmp/landing-dev.log 2>&1 &

    echo -e "${GREEN}Landing page restarted. Check /tmp/landing-dev.log for logs.${NC}"
}

case "$1" in
    status)
        show_status
        ;;
    primary)
        switch_to_primary
        ;;
    backup)
        switch_to_backup
        ;;
    health)
        health_check
        ;;
    auto)
        auto_failover
        ;;
    restart)
        restart_services
        ;;
    *)
        echo "WWAI-Macro Failover Script"
        echo ""
        echo "Usage: $0 {status|primary|backup|health|auto|restart}"
        echo ""
        echo "Commands:"
        echo "  status   - Show current server configuration"
        echo "  primary  - Switch to primary server (96)"
        echo "  backup   - Switch to backup server (97)"
        echo "  health   - Check health of both servers"
        echo "  auto     - Auto failover (switch if primary is down)"
        echo "  restart  - Restart landing page service"
        echo ""
        show_status
        ;;
esac
