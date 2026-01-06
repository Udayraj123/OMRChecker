#!/bin/bash

# Test script for Option C deliverables
# Run this to verify both solutions are working

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                 TESTING OPTION C DELIVERABLES                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

pass() {
    echo -e "${GREEN}✓${NC} $1"
}

fail() {
    echo -e "${RED}✗${NC} $1"
    exit 1
}

info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PART 1: Testing Fixed Vanilla JS Version"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 1: Check viewer.html exists and has fixes
info "Checking viewer.html..."
if [ ! -f "src/processors/visualization/templates/viewer.html" ]; then
    fail "viewer.html not found"
fi
pass "viewer.html exists"

# Test 2: Check for height fix in CSS
if grep -q "min-height: 400px" "src/processors/visualization/templates/viewer.html"; then
    pass "CSS height fix applied"
else
    fail "CSS height fix missing"
fi

# Test 3: Check for error handling
if grep -q "console.error('vis.js library not loaded')" "src/processors/visualization/templates/viewer.html"; then
    pass "Error handling added"
else
    fail "Error handling missing"
fi

# Test 4: Generate visualization
info "Generating test visualization..."
rm -rf outputs/test_viz
uv run python -m src.utils.visualization_runner \
    --input "samples/1-mobile-camera/MobileCamera/sheet1.jpg" \
    --template "samples/1-mobile-camera/template.json" \
    --output outputs/test_viz \
    > /dev/null 2>&1

if [ -f outputs/test_viz/*.html ]; then
    pass "Visualization generated successfully"
else
    fail "Visualization generation failed"
fi

# Test 5: Check HTML structure
HTML_FILE=$(find outputs/test_viz -name "*.html" | head -1)
if grep -q "workflow-graph" "$HTML_FILE"; then
    pass "HTML structure intact"
else
    fail "HTML structure broken"
fi

# Test 6: Check session data embedded
if grep -q "sessionData" "$HTML_FILE"; then
    pass "Session data embedded"
else
    fail "Session data missing"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PART 2: Testing React Migration Deliverables"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 7: Check migration guide exists
info "Checking documentation..."
if [ ! -f "docs/react-migration-guide.md" ]; then
    fail "Migration guide not found"
fi
pass "Migration guide exists"

# Test 8: Check guide has all phases
PHASES=$(grep -c "### Phase [0-9]:" "docs/react-migration-guide.md" || true)
if [ "$PHASES" -ge 7 ]; then
    pass "All 7 phases documented ($PHASES found)"
else
    fail "Missing phases (expected 7, found $PHASES)"
fi

# Test 9: Check React scaffold directory
info "Checking React scaffold..."
if [ ! -d "workflow-viz-app" ]; then
    fail "React scaffold directory not found"
fi
pass "React scaffold directory exists"

# Test 10: Check package.json
if [ ! -f "workflow-viz-app/package.json" ]; then
    fail "package.json not found"
fi
pass "package.json exists"

# Test 11: Check dependencies
if grep -q "react" "workflow-viz-app/package.json"; then
    pass "React dependency listed"
else
    fail "React dependency missing"
fi

if grep -q "react-flow-renderer" "workflow-viz-app/package.json"; then
    pass "React Flow dependency listed"
else
    fail "React Flow dependency missing"
fi

# Test 12: Check config files
for file in vite.config.ts tsconfig.json tailwind.config.js; do
    if [ -f "workflow-viz-app/$file" ]; then
        pass "$file exists"
    else
        fail "$file not found"
    fi
done

# Test 13: Check summary docs
info "Checking summary documentation..."
for doc in VISUALIZATION_SOLUTION_SUMMARY.md OPTION_C_COMPLETE.md; do
    if [ -f "$doc" ]; then
        pass "$doc exists"
    else
        fail "$doc not found"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

pass "All tests passed!"
echo ""
echo -e "${GREEN}✅ Vanilla JS Version:${NC} Fixed and working"
echo -e "${GREEN}✅ React Migration:${NC} Complete guide + scaffold"
echo ""
echo "Next Steps:"
echo "  1. Test vanilla: open $HTML_FILE"
echo "  2. Read guide: docs/react-migration-guide.md"
echo "  3. Try React: cd workflow-viz-app && npm install"
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                       🎉 ALL SYSTEMS GO! 🎉                          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

