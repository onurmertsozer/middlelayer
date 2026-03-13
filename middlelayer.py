"""
MiddleLayer — Ops Intelligence for Fintech
==========================================
Automated PnL calculation, anomaly detection,
and AI-powered explanations via Claude API.
"""

import urllib.request
import csv
import json
import os
import argparse
from datetime import datetime
import anthropic

# ── Configuration ─────────────────────────────────────────────────────────────

FX_RATES_TO_EUR = {
    "EUR": 1.0, "USD": 0.92, "GBP": 1.17, "SEK": 0.088,
    "NOK": 0.086, "DKK": 0.134, "CHF": 1.05, "PLN": 0.23,
}

ANOMALY_RULES = {
    "large_loss_eur":       -5000,
    "large_gain_eur":        8000,
    "price_deviation_pct":    5.0,
    "late_hour_threshold":      22,
    "early_hour_threshold":      6,
    "consecutive_failures":      2,
}

# ── Data Loading ───────────────────────────────────────────────────────────────

def load_trades(filepath: str) -> list[dict]:
    trades = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append({
                "id":           row["trade_id"],
                "timestamp":    row["timestamp"],
                "instrument":   row["instrument"],
                "type":         row["trade_type"],
                "currency":     row["currency"],
                "quantity":     float(row["quantity"]),
                "entry_price":  float(row["entry_price"]),
                "market_price": float(row["market_price"]),
                "status":       row["status"],
                "desk":         row["desk"],
                "counterparty": row["counterparty"],
            })
    return trades

# ── PnL Engine ────────────────────────────────────────────────────────────────

def calculate_pnl(trade: dict) -> dict:
    qty, entry, mkt = trade["quantity"], trade["entry_price"], trade["market_price"]
    fx = FX_RATES_TO_EUR.get(trade["currency"], 1.0)

    raw_pnl = (mkt - entry) * qty if trade["type"] == "BUY" else (entry - mkt) * qty
    pnl_eur = raw_pnl * fx
    price_deviation_pct = abs((mkt - entry) / entry * 100) if entry != 0 else 0

    return {
        **trade, "pnl_local": round(raw_pnl, 2), "pnl_eur": round(pnl_eur, 2),
        "price_deviation_pct": round(price_deviation_pct, 2), "fx_rate": fx,
    }

def run_pnl_engine(trades: list[dict]) -> list[dict]:
    return [calculate_pnl(t) if t["status"] == "SETTLED" else {**t, "pnl_local": 0, "pnl_eur": 0, "price_deviation_pct": 0, "fx_rate": 0} for t in trades]

# ── Anomaly Detection ─────────────────────────────────────────────────────────

def detect_anomalies(trades: list[dict]) -> list[dict]:
    flagged = []
    failure_counts = {}

    for t in trades:
        flags = []
        if t["pnl_eur"] < ANOMALY_RULES["large_loss_eur"]: flags.append(f"LARGE_LOSS: €{t['pnl_eur']:,.0f}")
        if t["pnl_eur"] > ANOMALY_RULES["large_gain_eur"]: flags.append(f"LARGE_GAIN: €{t['pnl_eur']:,.0f}")
        if t["price_deviation_pct"] > ANOMALY_RULES["price_deviation_pct"]: flags.append(f"PRICE_DEVIATION: {t['price_deviation_pct']:.1f}%")
        
        try:
            dt = datetime.fromisoformat(t["timestamp"])
            if dt.hour >= ANOMALY_RULES["late_hour_threshold"] or dt.hour < ANOMALY_RULES["early_hour_threshold"]:
                flags.append(f"OFF_HOURS: {dt.hour:02d}:00")
        except ValueError:
            pass

        if t["status"] == "FAILED":
            key = t["instrument"]
            failure_counts[key] = failure_counts.get(key, 0) + 1
            if failure_counts[key] >= ANOMALY_RULES["consecutive_failures"]:
                flags.append(f"CONSECUTIVE_FAILURES: {failure_counts[key]}x on {key}")

        if flags:
            flagged.append({**t, "flags": flags, "flag_count": len(flags)})

    return sorted(flagged, key=lambda x: x["flag_count"], reverse=True)

# ── Claude API — AI Explanations ──────────────────────────────────────────────

def get_ai_explanation(flagged_trades: list[dict], summary: dict) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "⚠️ ANTHROPIC_API_KEY environment variable not set. Skipping AI analysis."

    client = anthropic.Anthropic(api_key=api_key)

    trades_summary = json.dumps([{
        "id": t["id"], "instrument": t["instrument"], "type": t["type"], "pnl_eur": t["pnl_eur"],
        "flags": t["flags"], "status": t["status"], "counterparty": t["counterparty"]
    } for t in flagged_trades[:10]], indent=2)

    prompt = f"""You are a senior Middle Office analyst at a fintech company. Review the flagged trades and provide a concise operations briefing.
PORTFOLIO SUMMARY:
- Total trades: {summary['total_trades']}
- Total PnL: €{summary['total_pnl_eur']:,.2f}
FLAGGED TRADES (top 10):
{trades_summary}

Write a structured briefing:
1. EXECUTIVE SUMMARY (2 sentences)
2. TOP CONCERNS (bullet list, max 3)
3. RECOMMENDED ACTIONS"""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

# ── Summary Statistics ─────────────────────────────────────────────────────────

def build_summary(trades: list[dict], flagged: list[dict]) -> dict:
    settled = [t for t in trades if t["status"] == "SETTLED"]
    failed  = [t for t in trades if t["status"] == "FAILED"]
    total_pnl = sum(t["pnl_eur"] for t in settled)

    by_desk = {}
    for t in settled:
        desk = t["desk"]
        by_desk[desk] = by_desk.get(desk, 0) + t["pnl_eur"]

    return {
        "total_trades":    len(trades),
        "settled":         len(settled),
        "failed":          len(failed),
        "total_pnl_eur":   round(total_pnl, 2),
        "flagged_count":   len(flagged),
        "by_desk":         {k: round(v, 2) for k, v in by_desk.items()},
        "generated_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# ── Slack Integration ─────────────────────────────────────────────────────────

def send_slack_alert(webhook_url: str, summary: dict, flagged: list[dict]):
    """
    Formats and sends a high-level briefing to a Slack channel via Incoming Webhooks.
    """
    if not webhook_url:
        return

    # Set status color: Red for anomalies, Green for a clean run
    color = "#991B1B" if summary["flagged_count"] > 0 else "#166534"
    
    # Format the top 5 anomalies for the Slack message
    flagged_text = ""
    for t in flagged[:5]:
        flags_str = ", ".join(t['flags'])
        flagged_text += f"• *{t['id']}* ({t['instrument']}): €{t['pnl_eur']:,.2f} ➔ {flags_str}\n"
    
    if not flagged_text:
        flagged_text = "All clear. No manual intervention required. ✅"

    # Prepare the Slack payload using Block Kit-style attachments
    slack_data = {
        "attachments": [
            {
                "color": color,
                "fallback": "MiddleLayer Daily Operations Briefing",
                "title": "📊 MiddleLayer Daily Ops Briefing",
                "text": (
                    f"*Total Trades Analyzed:* {summary['total_trades']}\n"
                    f"*Total PnL:* €{summary['total_pnl_eur']:,.2f}\n"
                    f"*Anomalies Flagged:* {summary['flagged_count']}\n\n"
                    f"*⚠️ Top Concerns:*\n{flagged_text}"
                ),
                "footer": "MiddleLayer Ops Intelligence",
                "ts": int(datetime.now().timestamp())
            }
        ]
    }

    # Execute the request using standard urllib
    req = urllib.request.Request(
        webhook_url, 
        data=json.dumps(slack_data).encode("utf-8"), 
        headers={"Content-Type": "application/json"}
    )
    
    try:
        urllib.request.urlopen(req)
        print("  ✅ Slack alert sent successfully!")
    except Exception as e:
        print(f"  ⚠️ Failed to send Slack alert: {e}")
# ── HTML Dashboard ────────────────────────────────────────────────────────────

def generate_html_report(trades: list[dict], flagged: list[dict], summary: dict, ai_analysis: str) -> str:
    pnl_color = "#166534" if summary["total_pnl_eur"] >= 0 else "#991B1B"
    desk_rows = "".join(f'<tr><td>{desk}</td><td style="color:{"#166534" if pnl>=0 else "#991B1B"};font-weight:500">€{pnl:,.2f}</td></tr>' for desk, pnl in summary["by_desk"].items())
    flagged_rows = "".join(f'<tr style="background:{"#FEE2E2" if t["flag_count"]>=2 else "#FEF3C7"}"><td><strong>{t["id"]}</strong></td><td>{t["instrument"]}</td><td>{t["type"]}</td><td>€{t["pnl_eur"]:,.2f}</td><td><span style="background:{"#991B1B" if t["flag_count"]>=2 else "#92400E"};color:white;padding:2px 8px;border-radius:4px;font-size:11px">{"HIGH" if t["flag_count"]>=2 else "MEDIUM"}</span></td><td style="font-size:12px;color:#374151">{" · ".join(t["flags"])}</td></tr>' for t in flagged[:15])

    ai_html = ai_analysis.replace("\n", "<br>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <style>
    body {{ font-family: -apple-system, sans-serif; background: #F8FAFC; color: #1E293B; font-size: 14px; margin: 0; padding: 0; }}
    .header {{ background: #1B3A6B; color: white; padding: 24px 32px; display: flex; justify-content: space-between; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-bottom: 24px; }}
    .metric {{ background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; text-align: center; }}
    .value {{ font-size: 24px; font-weight: bold; }}
    .card {{ background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
    table {{ width: 100%; border-collapse: collapse; text-align: left; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #E2E8F0; }}
    th {{ background: #F1F5F9; font-size: 12px; text-transform: uppercase; }}
    .ai-box {{ background: #EFF6FF; border-left: 4px solid #3B82F6; padding: 16px; line-height: 1.6; }}
  </style>
</head>
<body>
<div class="header">
  <div><h1 style="margin:0;">MiddleLayer</h1><p style="margin-top:5px; opacity:0.8;">Ops Intelligence for Fintech</p></div>
  <div style="text-align:right">Generated: {summary['generated_at']}</div>
</div>
<div class="container">
  <div class="metrics">
    <div class="metric"><div class="value" style="color:{pnl_color}">€{summary['total_pnl_eur']:,.0f}</div>Total PnL</div>
    <div class="metric"><div class="value">{summary['total_trades']}</div>Total Trades</div>
    <div class="metric"><div class="value" style="color:#166534">{summary['settled']}</div>Settled</div>
    <div class="metric"><div class="value" style="color:#991B1B">{summary['failed']}</div>Failed</div>
    <div class="metric"><div class="value" style="color:#92400E">{summary['flagged_count']}</div>Flagged</div>
  </div>
  <div class="card"><h2 style="margin-top:0;">AI Operations Briefing</h2><div class="ai-box">{ai_html}</div></div>
  <div class="card"><h2 style="margin-top:0;">Flagged Trades</h2><table><thead><tr><th>ID</th><th>Instrument</th><th>Type</th><th>PnL (EUR)</th><th>Severity</th><th>Flags</th></tr></thead><tbody>{flagged_rows}</tbody></table></div>
  <div class="card"><h2 style="margin-top:0;">PnL by Desk</h2><table><thead><tr><th>Desk</th><th>PnL (EUR)</th></tr></thead><tbody>{desk_rows}</tbody></table></div>
</div>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MiddleLayer — Ops Intelligence")
    parser.add_argument("--input",  default="trades.csv",   help="Trade CSV file path")
    parser.add_argument("--output", default="report.html",  help="HTML report output path")
    # New argument for Slack integration:
    parser.add_argument("--slack-webhook", default=None,    help="Slack Incoming Webhook URL")
    args = parser.parse_args()

    print("MiddleLayer — starting analysis...")
    
    # Load and process trades 
    trades  = load_trades(args.input)
    trades  = run_pnl_engine(trades)
    flagged = detect_anomalies(trades)
    summary = build_summary(trades, flagged)

    # Trigger Slack Alert if a URL is provided 
    if args.slack_webhook:
        print("  Sending notification to Slack...")
        send_slack_alert(args.slack_webhook, summary, flagged)

    # Continue to AI analysis and HTML report generation 
    print("  Calling Claude API for AI analysis...")
    ai_analysis = get_ai_explanation(flagged, summary)

    html = generate_html_report(trades, flagged, summary, ai_analysis)
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"✅ Report successfully generated: {args.output}")
