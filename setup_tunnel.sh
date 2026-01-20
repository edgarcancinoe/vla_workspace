#!/bin/bash
# SSH Tunnel Setup Script
# This creates an SSH tunnel to forward local port 8080 to the policy server on flexo

echo "Setting up SSH tunnel to flexo cluster..."
echo "   Local port 8080 -> flexo:8080"
echo ""
echo "Press Ctrl+C to close the tunnel when done"
echo ""

# Use the SSH config entry for flexo
# -L 8080:localhost:8080 means: forward my local port 8080 to flexo's localhost:8080
# -N means: don't execute a remote command, just forward the port
# -v means: verbose (optional, remove if you want less output)

ssh -L 8080:localhost:8080 -N vicos-flexo
