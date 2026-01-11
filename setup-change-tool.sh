#!/bin/bash
# Setup script for Change Propagation Tool

set -e

echo "🚀 Setting up OMRChecker Change Propagation Tool..."

# Check for pnpm
if ! command -v pnpm &> /dev/null; then
    echo "❌ pnpm not found. Installing..."
    npm install -g pnpm
fi

# Navigate to change tool directory
cd "$(dirname "$0")/change-propagation-tool"

echo "📦 Installing dependencies..."
pnpm install

echo "✅ Setup complete!"
echo ""
echo "To start the development server:"
echo "  cd change-propagation-tool"
echo "  pnpm dev"
echo ""
echo "To build for production:"
echo "  cd change-propagation-tool"
echo "  pnpm build"
echo ""

