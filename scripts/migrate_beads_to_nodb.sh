#!/bin/bash
# Migrate beads from dolt mode to no-db (JSONL) mode

set -e

echo "=== Migrating beads from dolt to no-db mode ==="

# Backup current state
echo "1. Creating backup..."
cp -r .beads .beads.backup.$(date +%Y%m%d_%H%M%S)

# Update metadata.json to jsonl mode
echo "2. Updating metadata.json..."
cat > .beads/metadata.json << 'EOF'
{
  "database": "jsonl",
  "jsonl_export": "issues.jsonl",
  "backend": "jsonl"
}
EOF

# Create empty issues.jsonl if it doesn't exist
echo "3. Ensuring issues.jsonl exists..."
touch .beads/issues.jsonl

# Ensure config.yaml has no-db: true
echo "4. Verifying config.yaml..."
if ! grep -q "no-db: true" .beads/config.yaml; then
    sed -i.bak 's/# no-db: false/no-db: true/' .beads/config.yaml
fi

# Test the configuration
echo "5. Testing beads..."
if bd list >/dev/null 2>&1; then
    echo "✅ Success! Beads is now running in no-db mode"
    echo "   - Backend: JSONL (no dolt dependency)"
    echo "   - Storage: .beads/issues.jsonl"
    echo "   - Backup: .beads.backup.*"
else
    echo "❌ Test failed. Restoring from backup..."
    rm -rf .beads
    mv .beads.backup.* .beads
    exit 1
fi

echo ""
echo "Done! You can now use bd commands without dolt."
echo "Try: bd list"
