# Integration recipes — prove-or-abstain as a workflow component

Copy-paste examples for embedding prove-or-abstain in production workflows.
All examples assume the API is running at `http://localhost:8000` or a deployed URL.

---

## Cron — scheduled autonomous checks every 10 minutes
> `integrations/cron.sh`

```bash
*/10 * * * * curl -sf -X POST https://your-api.example.com/investigate/check | python3 -c "
import sys,json
d=json.load(sys.stdin)
if d['verdict']=='ASSERT_ACTED':
    print(f'AUTOPILOT: {sum(1 for p in d[\"panels\"] if p[\"action\"][\"kind\"]==\"EXECUTE\")} actions taken')
"
```

---

## n8n — HTTP Request node
> Import `integrations/n8n.json` into n8n as a new workflow, or create a node manually:

**HTTP Request node configuration:**
```
Method: POST
URL: https://your-api.example.com/investigate/upload
Body Type: Form-Data
  - baseline: binary data from previous node
  - current: binary data from previous node
  - sum_metrics: revenue,cost
  - autopilot: true
```

**Processing the response (Function node):**
```js
const body = $input.first().json;
if (body.verdict === 'ASSERT' && body.confidence >= 0.70) {
  return { action: 'EXECUTE', cause: body.root_cause, detail: body.action.detail };
}
return { action: 'ESCALATE', verdict: body.verdict };
```

---

## GitHub Actions — investigate on data change
> `integrations/github-actions.yml`

```yaml
name: investigate metric change
on:
  push:
    paths: ['data/*.csv']
jobs:
  investigate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run investigation
        run: |
          pip install -r requirements.txt
          python -m prove_or_abstain.benchmark
      - name: Notify on failure
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {"text": "⚠️ Benchmark accuracy dropped below 100%"}
```

---

## Slack bot — slash command to investigate
> `integrations/slack-bot.sh`

```bash
#!/bin/bash
# Deploy as a Slack slash command handler.
# Slack calls this URL when someone types /investigate metric_name

METRIC=$1

# Fetch data from your warehouse (example)
curl -sf "https://your-warehouse/api/metrics/$METRIC/baseline" -o /tmp/base.csv
curl -sf "https://your-warehouse/api/metrics/$METRIC/current" -o /tmp/curr.csv

# Investigate
RESULT=$(curl -sf -X POST http://localhost:8000/investigate/upload \
  -F "baseline=@/tmp/base.csv" \
  -F "current=@/tmp/curr.csv" \
  -F "autopilot=true")

VERDICT=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['verdict'])")

echo "{\"text\": \"Investigation complete: $VERDICT for $METRIC\"}"
```

---

## cURL — one-liners for common tasks

```bash
# Investigate a named panel
curl -s -X POST http://localhost:8000/investigate -H 'content-type: application/json' -d '{"panel":"clean","autopilot":true}'

# Investigate from SQL
curl -s -X POST http://localhost:8000/investigate/sql -H 'content-type: application/json' -d '{
  "dsn": "postgresql://user:pass@host/db",
  "baseline_query": "SELECT metric, segment, device, n, c FROM stats WHERE period=''2024-01''",
  "current_query":  "SELECT metric, segment, device, n, c FROM stats WHERE period=''2024-02''"
}'

# Check dashboard
curl -s http://localhost:8000/dashboard

# Resolve an alert
curl -s -X POST http://localhost:8000/executions/conversion:segment=paid/resolve \
  -H 'content-type: application/json' -d '{"id":"conversion:segment=paid","resolved_by":"human"}'
```

---

## MCP — Qwen Cloud agent

```bash
# Start the MCP server (stdio — Qwen Cloud connects automatically)
python mcp_server.py

# SSE transport for local testing
python mcp_server.py --port 8080
```

A Qwen Cloud agent can then call:
- `investigate_scenario("clean")` — investigate a built-in scenario
- `investigate_sql(dsn, query1, query2)` — query a live database
- `autonomous_check()` — run all panels with autopilot
- `get_dashboard()` — view active alerts
- `resolve_alert(id)` — human-in-the-loop resolution
