#!/bin/bash
# Run prove-or-abstain's autonomous check on a schedule.
# Usage: */10 * * * * /path/to/integrations/autonomous-check.sh
#
# Set these env vars on your server:
#   PROVEO_URL  — where the API is running (default http://localhost:8000)
#   WEBHOOK_URL — optional Slack/Discord URL for alerts

set -euo pipefail

BASE_URL="${PROVEO_URL:-http://localhost:8000}"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Running autonomous check..."
RESULT=$(curl -sf -X POST "${BASE_URL}/investigate/check" || echo '{"error":"API unreachable"}')

if echo "$RESULT" | grep -q '"ASSERT_ACTED"'; then
  echo "  → ASSERT_ACTED — autopilot took action"
  echo "$RESULT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for p in d.get('panels', []):
    if p.get('action', {}).get('kind') == 'EXECUTE':
        print(f'  ▸ {p[\"panel\"]}: {p[\"root_cause\"][\"dimension\"]}={p[\"root_cause\"][\"segment\"]} (conf={p[\"confidence\"]:.2f})')" 2>/dev/null || true
else
  echo "  → NO_ANOMALY — nothing to act on"
fi
