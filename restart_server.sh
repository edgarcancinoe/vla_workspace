#!/bin/bash
# Quick script to restart the server on flexo after pulling updates

echo "Pulling latest changes..."
git pull

echo ""
echo "Restarting policy server..."
echo "   Press Ctrl+C to stop the server"
echo ""

./run_server.sh
