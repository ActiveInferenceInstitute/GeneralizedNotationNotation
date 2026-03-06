#!/usr/bin/env bash
# scripts/count_tests.sh — Reproducible test count validation
# Compares collected test count against the documented minimum (1,522).
# Exit 1 if drift exceeds 5%.
set -euo pipefail

DOCUMENTED_MIN=1522
DRIFT_THRESHOLD=5  # percent

cd "$(git rev-parse --show-toplevel)"

echo "📊 Counting tests..."
COUNT=$(python -m pytest --co -q 2>/dev/null | tail -1 | grep -oE '^[0-9]+')

if [ -z "$COUNT" ]; then
    echo "❌ Could not determine test count"
    exit 1
fi

echo "   Collected: $COUNT tests"
echo "   Documented minimum: $DOCUMENTED_MIN"

DRIFT=$(python3 -c "
import sys
count, doc = int(sys.argv[1]), int(sys.argv[2])
drift = abs(count - doc) / doc * 100
print(f'{drift:.1f}')
" "$COUNT" "$DOCUMENTED_MIN")

echo "   Drift: ${DRIFT}%"

if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= $DRIFT_THRESHOLD else 1)" "$DRIFT"; then
    echo "✅ Test count within ${DRIFT_THRESHOLD}% of documented minimum"
else
    echo "❌ Test count drifted ${DRIFT}% (threshold: ${DRIFT_THRESHOLD}%)"
    echo "   Update TO-DO.md or add/restore tests"
    exit 1
fi
