import os
import json
import psycopg2
import anthropic
import requests
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# CONFIG — fill these in
# ─────────────────────────────────────────────

DB_CONFIG = {
    "host":     "YOUR_DB_HOST",
    "port":     5432,
    "dbname":   "YOUR_DB_NAME",
    "user":     "YOUR_DB_USER",
    "password": "YOUR_DB_PASSWORD",
}

# The table where your job writes its output
JOB_OUTPUT_TABLE = "job_outputs"

# Number of historical rows to give Claude for pattern learning
HISTORY_LIMIT = 50

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY")
SLACK_WEBHOOK_URL = "YOUR_SLACK_WEBHOOK_URL"  # https://hooks.slack.com/services/...


# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def fetch_latest_output(conn):
    """Fetch the most recent job output row."""
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT *
            FROM {JOB_OUTPUT_TABLE}
            ORDER BY created_at DESC
            LIMIT 1
        """)
        cols = [desc[0] for desc in cur.description]
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip(cols, row))


def fetch_historical_outputs(conn, exclude_id=None):
    """Fetch recent historical outputs for Claude to learn patterns from."""
    with conn.cursor() as cur:
        query = f"""
            SELECT *
            FROM {JOB_OUTPUT_TABLE}
            {"WHERE id != %s" if exclude_id else ""}
            ORDER BY created_at DESC
            LIMIT {HISTORY_LIMIT}
        """
        cur.execute(query, (exclude_id,) if exclude_id else ())
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]


# ─────────────────────────────────────────────
# AGENT — Claude does the reasoning
# ─────────────────────────────────────────────

def serialize(data):
    """Make DB rows JSON-serializable (handles datetime, etc.)."""
    def convert(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)
    return json.dumps(data, default=convert, indent=2)


def run_agent(latest_output, historical_outputs):
    """
    Send historical + latest output to Claude.
    Claude reasons about whether the latest output is anomalous.
    Returns a dict: { ok: bool, verdict: str, reason: str }
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""
You are a data quality monitoring agent. Your job is to analyze job outputs 
stored in a database and detect anomalies.

Here are the last {len(historical_outputs)} historical job outputs (these represent 
normal, expected behavior — learn the patterns from them):

<historical_outputs>
{serialize(historical_outputs)}
</historical_outputs>

Here is the LATEST job output that just ran:

<latest_output>
{serialize(latest_output)}
</latest_output>

Analyze the latest output against the historical patterns. Look for things like:
- Unexpected changes in record counts, values, or structure
- Missing or null fields that were previously populated
- Status changes (e.g. success vs failure rates)
- Unusual timing gaps
- Any metric that deviates significantly from the historical norm
- Patterns that suggest silent failures (e.g. status=success but output looks wrong)

Respond ONLY with a JSON object in this exact format (no markdown, no explanation outside JSON):
{{
  "ok": true or false,
  "verdict": "one-line summary of your finding",
  "reason": "detailed explanation of what you found and why it is or isn't anomalous"
}}
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if Claude didn't return clean JSON
        return {
            "ok": False,
            "verdict": "Agent response parse error",
            "reason": raw
        }


# ─────────────────────────────────────────────
# ALERTING
# ─────────────────────────────────────────────

def send_slack_alert(result, latest_output):
    """Send a Slack alert with Claude's finding."""
    job_id = latest_output.get("id", "unknown")
    created_at = latest_output.get("created_at", "unknown")

    color = "#36a64f" if result["ok"] else "#ff0000"
    status_emoji = "✅" if result["ok"] else "🚨"

    payload = {
        "attachments": [
            {
                "color": color,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{status_emoji} Job Output Verification — {'PASSED' if result['ok'] else 'ANOMALY DETECTED'}"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Job ID:*\n{job_id}"},
                            {"type": "mrkdwn", "text": f"*Run At:*\n{created_at}"},
                        ]
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Verdict:*\n{result['verdict']}"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Reason:*\n{result['reason']}"}
                    }
                ]
            }
        ]
    }

    resp = requests.post(SLACK_WEBHOOK_URL, json=payload)
    if resp.status_code != 200:
        print(f"[WARN] Slack alert failed: {resp.status_code} {resp.text}")
    else:
        print(f"[INFO] Slack alert sent — {'OK' if result['ok'] else 'ANOMALY'}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"[{datetime.now(timezone.utc).isoformat()}] Running verification agent...")

    conn = get_db_connection()
    try:
        latest = fetch_latest_output(conn)
        if latest is None:
            print("[WARN] No job outputs found in DB. Skipping.")
            return

        history = fetch_historical_outputs(conn, exclude_id=latest.get("id"))

        if len(history) < 5:
            print(f"[INFO] Only {len(history)} historical rows — agent will have limited pattern data.")

        print(f"[INFO] Fetched latest output (id={latest.get('id')}) and {len(history)} historical rows.")
        print("[INFO] Sending to Claude for analysis...")

        result = run_agent(latest, history)

        print(f"[INFO] Agent verdict: {result['verdict']}")
        print(f"[INFO] OK: {result['ok']}")
        print(f"[INFO] Reason: {result['reason']}")

        # Always send to Slack (you can change this to only alert on anomalies)
        if not result["ok"]:
            send_slack_alert(result, latest)
        else:
            print("[INFO] Output looks normal. No alert sent.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
