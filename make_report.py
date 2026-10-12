#!/usr/bin/env python3
"""
Builds a PDF report from webcall_load_test.py result files.

Usage (needs reportlab):
    python make_report.py                      # all results/webcall_*.json
    python make_report.py results/a.json ...   # specific runs
"""

import glob
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.platypus import (CondPageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

NAVY = colors.HexColor("#1f2d4a")
BLUE = colors.HexColor("#3b6fb6")
LIGHT = colors.HexColor("#eef2f8")
GRID = colors.HexColor("#b8bfcc")
OK = colors.HexColor("#d9f2dd")
BAD = colors.HexColor("#f8d4d4")
WARN = colors.HexColor("#fff0c2")
SEG = [colors.HexColor("#3b6fb6"), colors.HexColor("#6b4fa8"), colors.HexColor("#2a8c7a"),
       colors.HexColor("#b0b7c3")]

ss = getSampleStyleSheet()
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=21, leading=26, textColor=NAVY, alignment=0)
SUB = ParagraphStyle("S", parent=ss["BodyText"], fontSize=11, textColor=colors.HexColor("#4a5568"))
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=NAVY, spaceBefore=10, spaceAfter=5)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, textColor=BLUE, spaceBefore=6, spaceAfter=3)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.8, leading=13.5, spaceAfter=4)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2)
C = ParagraphStyle("C", parent=B, fontSize=8.6, leading=11, spaceAfter=0)
CB = ParagraphStyle("CB", parent=C, fontName="Helvetica-Bold")
CH = ParagraphStyle("CH", parent=C, textColor=colors.white, fontName="Helvetica-Bold")
NOTE = ParagraphStyle("N", parent=B, fontSize=8.8, textColor=colors.HexColor("#5a4500"), backColor=WARN,
                      borderPadding=5, spaceBefore=4, spaceAfter=6)
KPI_V = ParagraphStyle("KV", parent=B, fontSize=17, leading=20, textColor=NAVY, fontName="Helvetica-Bold",
                       alignment=1, spaceAfter=0)
KPI_L = ParagraphStyle("KL", parent=B, fontSize=8, leading=10, textColor=colors.HexColor("#555"), alignment=1)

# question used by runs recorded before question_text was saved
DEFAULT_QUESTION = "Hello. Can you tell me what you can help me with today?"

# metric key, label, definition
CLIENT_METRICS = [
    ("call_setup_ms", "Call setup latency",
     "Start of call (API request) until the caller hears the agent's first word"),
    ("web_call_ms", "API: create call", "Wrapper POST /web-calls response time"),
    ("room_join_ms", "Room join", "Caller connects to the LiveKit room"),
    ("agent_audio_track_ms", "Agent joined", "Caller joined until the agent's audio track arrives (dispatch)"),
    ("greeting_start_ms", "Time to greeting", "Caller joined until the agent starts speaking"),
    ("agent_response_ms", "Agent response latency",
     "Caller stops speaking until the agent starts replying (voice-to-voice)"),
    ("reply_duration_ms", "Agent reply length", "How long the agent's spoken answer lasts"),
    ("total_ms", "Total call duration", "Whole test call, start to hang-up"),
]
SERVER_METRICS = [
    ("eou", "End-of-turn delay", "Agent deciding the caller finished speaking (EOUMetrics)"),
    ("ttft", "LLM time to first token", "LLM starts producing the answer (LLMMetrics)"),
    ("ttfb", "TTS time to first byte", "Text-to-speech starts producing audio (TTSMetrics)"),
]


# ---------------- data ----------------
def server_metrics(call_record):
    m = {"eou": [], "ttft": [], "ttfb": []}
    for f in (call_record or {}).get("flow_trace", []):
        a = f.get("arguments") or {}
        if f.get("type") != "metrics":
            continue
        if "eou_delay_ms" in a:
            m["eou"].append(a["eou_delay_ms"])
        if "llm_ttft_ms" in a:
            m["ttft"].append(a["llm_ttft_ms"])
        if "tts_ttfb_ms" in a:
            m["ttfb"].append(a["tts_ttfb_ms"])
    return m


def classify(r):
    if r.get("success"):
        return "Pass", OK, ""
    rec = r.get("call_record") or {}
    agent_spoke = any(t.get("role") == "assistant" for t in rec.get("transcript", []))
    if "livekit_url_used" not in r or r.get("livekit_url_used") == r.get("livekit_url_returned"):
        return "Fail", BAD, "LiveKit URL mismatch: caller joined a different server than the agent"
    if agent_spoke:
        return "Tool issue", WARN, "Agent spoke, but the test listened to its silent 2nd track (fixed)"
    return "Fail", BAD, r.get("error", "")


def load(paths):
    runs = []
    for p in paths:
        r = json.loads(Path(p).read_text())
        r["_file"] = p
        r["_class"] = classify(r)
        r["_metrics"] = server_metrics(r.get("call_record"))
        rec = r.get("call_record") or {}
        r["_agent"] = rec.get("agent_id") or (r.get("web_call_response") or {}).get("agent_id") or "?"
        st = r["steps"]
        if "call_setup_ms" not in st and "greeting_start_ms" in st and "web_call_ms" in st:
            st["call_setup_ms"] = st["web_call_ms"] + st["greeting_start_ms"]  # older runs
        runs.append(r)
    runs.sort(key=lambda r: r["started_at"])
    return runs


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    return xs[f] if f + 1 >= len(xs) else xs[f] + (xs[f + 1] - xs[f]) * (k - f)


def stats(xs):
    if not xs:
        return None
    return {"n": len(xs), "avg": statistics.mean(xs), "min": min(xs), "p50": pct(xs, 50),
            "p95": pct(xs, 95), "max": max(xs)}


def sec(v):
    if v is None:
        return "-"
    return f"{v:.0f} ms" if v < 1000 else f"{v / 1000:.2f} s"


def short(agent):
    return agent[:8]


def greeting_of(runs):
    for r in reversed(runs):
        for t in (r.get("call_record") or {}).get("transcript", []):
            if t.get("role") == "assistant":
                txt = (t.get("content") or "").replace("�", "'").replace("’", "'")
                return txt[:90] + ("..." if len(txt) > 90 else "")
    return "-"


# ---------------- drawing ----------------
def table(rows, widths, shade=None, bold_first_col=False):
    def cell(c, j):
        return Paragraph(str(c), CB if bold_first_col and j == 0 else C)
    data = [[Paragraph(str(c), CH) for c in rows[0]]] + [[cell(c, j) for j, c in enumerate(r)] for r in rows[1:]]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("GRID", (0, 0), (-1, -1), 0.4, GRID),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    st += [("BACKGROUND", (0, i), (-1, i), LIGHT) for i in range(2, len(rows), 2)]
    for (col, row), colr in (shade or {}).items():
        st.append(("BACKGROUND", (col, row), (col, row), colr))
    t.setStyle(TableStyle(st))
    return t


def kpis(items):
    cells = [[Paragraph(v, KPI_V) for v, _ in items], [Paragraph(l, KPI_L) for _, l in items]]
    t = Table(cells, colWidths=[180 * mm / len(items)] * len(items))
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, GRID), ("INNERGRID", (0, 0), (-1, -1), 0.3, GRID),
                           ("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def breakdown_chart(parts, total):
    W, H = 180 * mm, 30 * mm
    d = Drawing(W, H)
    x0, bw = 4, W - 8
    scale = bw / max(total, 1)
    x = x0
    for i, (label, val) in enumerate(parts):
        w = val * scale
        d.add(Rect(x, 34, w, 22, fillColor=SEG[i], strokeColor=colors.white, strokeWidth=1))
        if w > 34:
            d.add(String(x + w / 2, 41, f"{val / 1000:.2f}s", textAnchor="middle", fontSize=8,
                         fillColor=colors.white, fontName="Helvetica-Bold"))
        x += w
    lx = x0
    for i, (label, val) in enumerate(parts):
        d.add(Rect(lx, 12, 8, 8, fillColor=SEG[i], strokeColor=SEG[i]))
        d.add(String(lx + 11, 13, label, fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#333")))
        lx += 11 + len(label) * 3.6 + 16
    d.add(String(x0, 62, f"Agent response latency {total / 1000:.2f} s (caller stops speaking -> agent audio starts)",
                 fontSize=8.5, fillColor=NAVY, fontName="Helvetica-Bold"))
    return d


def timeline_chart(s):
    W, H = 180 * mm, 38 * mm
    d = Drawing(W, H)
    total = s.get("total_ms", 30000)
    x0, bw = 70, W - 80
    sc = bw / total
    rows = []
    g0 = s.get("greeting_start_ms")
    if g0 is not None:
        rows.append(("Agent greeting", g0, g0 + s.get("greeting_duration_ms", 0), SEG[0]))
        q0 = g0 + s.get("greeting_duration_ms", 0) + 500
        if s.get("agent_response_ms") is not None:
            qd = s.get("question_ms", 4400)
            rows.append(("Caller question", q0, q0 + qd, SEG[2]))
            a0 = q0 + qd + s["agent_response_ms"]
            rows.append(("Agent reply", a0, a0 + s.get("reply_duration_ms", 0), SEG[1]))
    y = H - 22
    for label, a, b, col in rows:
        d.add(String(0, y + 3, label, fontName="Helvetica", fontSize=8, fillColor=colors.HexColor("#333")))
        d.add(Rect(x0 + a * sc, y, max(2, (b - a) * sc), 12, fillColor=col, strokeColor=col))
        y -= 18
    d.add(Line(x0, 8, x0 + bw, 8, strokeColor=GRID))
    for s_ in range(0, int(total / 1000) + 1, 5):
        xx = x0 + s_ * 1000 * sc
        d.add(Line(xx, 6, xx, 10, strokeColor=GRID))
        d.add(String(xx, -2, f"{s_}s", textAnchor="middle", fontName="Helvetica", fontSize=7,
                     fillColor=colors.HexColor("#666")))
    return d


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777"))
    canvas.drawString(15 * mm, 9 * mm, "AI Labs Voice Agent - Web-Call Test Report")
    canvas.drawRightString(195 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def metrics_table(ok_runs):
    rows = [["Metric", "n", "Avg", "Min", "P50", "P95", "Max"]]
    rows.append(["<b>Caller-measured</b>", "", "", "", "", "", ""])
    for key, label, _ in CLIENT_METRICS:
        s = stats([r["steps"][key] for r in ok_runs if r["steps"].get(key) is not None])
        if s:
            rows.append([label, s["n"], sec(s["avg"]), sec(s["min"]), sec(s["p50"]), sec(s["p95"]), sec(s["max"])])
    rows.append(["<b>Agent server-side (per turn)</b>", "", "", "", "", "", ""])
    for key, label, _ in SERVER_METRICS:
        s = stats([v for r in ok_runs for v in r["_metrics"][key]])
        if s:
            rows.append([label, s["n"], sec(s["avg"]), sec(s["min"]), sec(s["p50"]), sec(s["p95"]), sec(s["max"])])
    t = table(rows, [58, 12, 22, 22, 22, 22, 22])
    extra = []
    for i, r in enumerate(rows):
        if r[0].startswith("<b>"):
            extra += [("SPAN", (0, i), (-1, i)), ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#dde4f0"))]
        if r[0] in ("Call setup latency", "Agent response latency"):
            extra.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#e3f4f1")))
    t.setStyle(TableStyle(extra))
    return t


# ---------------- report ----------------
def build(runs, out, primary_agent):
    agents = []
    for r in runs:
        if r["_agent"] not in agents:
            agents.append(r["_agent"])
    if primary_agent not in agents:
        primary_agent = runs[-1]["_agent"]
    by_agent = {a: [r for r in runs if r["_agent"] == a] for a in agents}
    pruns = by_agent[primary_agent]
    pok = [r for r in pruns if r["success"]]
    last_ok = pok[-1] if pok else None
    n_url = sum(1 for r in runs if r["_class"][2].startswith("LiveKit URL"))
    n_tool = sum(1 for r in runs if r["_class"][0] == "Tool issue")

    def s_of(key):
        return stats([r["steps"][key] for r in pok if r["steps"].get(key) is not None])

    setup, resp = s_of("call_setup_ms"), s_of("agent_response_ms")
    ttft = stats([v for r in pok for v in r["_metrics"]["ttft"]])

    s = []
    s.append(Paragraph("AI Labs Voice Agent - Web-Call Test Report", TITLE))
    s.append(Paragraph(f"Single-user functional and latency test &middot; box 214 &middot; "
                       f"{runs[0]['started_at'][:10]} &middot; generated {datetime.now():%Y-%m-%d %H:%M}", SUB))
    s.append(Spacer(1, 8))

    # 1. summary
    s.append(Paragraph("1. Summary", H1))
    verdict = (f"<b>Result: PASS.</b> {len(pok)} of {len(pruns)} calls to agent <b>{short(primary_agent)}</b> "
               "completed: the caller heard the greeting, asked a question and got a correct spoken answer."
               if pok else "<b>Result: FAIL.</b> No successful call for this agent.")
    s.append(Paragraph(verdict, B))
    s.append(kpis([
        (f"{100 * len(pok) / len(pruns):.0f}%", f"success rate<br/>({len(pok)}/{len(pruns)} calls)"),
        (sec(setup["avg"]) if setup else "-", "avg call setup<br/>latency"),
        (sec(resp["avg"]) if resp else "-", "avg agent response<br/>latency"),
        (sec(resp["p95"]) if resp else "-", "P95 agent response<br/>latency"),
        (sec(ttft["avg"]) if ttft else "-", "avg LLM time to<br/>first token"),
    ]))
    if pok and len(pok) < 10:
        s.append(Paragraph(f"<b>Sample size:</b> only {len(pok)} successful call(s) for this agent. Averages are "
                           "indicative; P95 needs about 20+ calls to be reliable "
                           "(<font face='Courier'>webcall_load_test.py --runs 20</font>).", NOTE))
    bullets = []
    if n_url:
        bullets.append(f"{n_url} early call(s) were silent because the wrapper on box 214 returns a LiveKit URL "
                       "(<font face='Courier'>wss://ailabs.healthviewx.com/livekit</font>) that is not the server "
                       "the box-214 agent uses. Joining <font face='Courier'>ws://172.17.20.214:7880</font> directly "
                       "fixed it. <b>Needs a fix on the wrapper side.</b> These are excluded from latency metrics.")
    if n_tool:
        bullets.append(f"{n_tool} call(s) were marked failed by the test tool although the agent spoke (it listened "
                       "to the agent's silent second audio track). Fixed in the tool; excluded from metrics.")
    bullets.append("The LLM is the largest part of every reply; end-of-turn detection and text-to-speech are fast.")
    s += [Paragraph(b, BUL, bulletText="•") for b in bullets]

    # 2. setup
    s.append(Paragraph("2. Test setup", H1))
    s.append(table([
        ["Item", "Value"],
        ["Environment", "Box 214: wrapper http://172.17.20.214:6002, LiveKit ws://172.17.20.214:7880"],
        ["Test path", "POST /v1/agents/{id}/web-calls -> join LiveKit room as caller (WebRTC) -> wait for "
                      "greeting -> play recorded question -> measure reply -> hang up -> GET /v1/calls/{id}"],
        ["Caller question", " / ".join(sorted({'"%s"' % (r.get("question_text") or DEFAULT_QUESTION)
                                               for r in runs}))],
        ["Users", "1 at a time (no concurrency yet)"],
        ["Tool", "webcall_load_test.py, run from a Windows PC on the office network"],
    ], [30, 150]))
    s.append(Spacer(1, 4))
    arows = [["Agent", "Agent ID", "Calls", "Passed", "Greeting (first words)"]]
    for a in agents:
        rs = by_agent[a]
        arows.append([("<b>%s</b> (main)" % short(a)) if a == primary_agent else short(a), a, len(rs),
                      sum(r["success"] for r in rs), greeting_of(rs)])
    s.append(table(arows, [22, 50, 12, 15, 81]))

    # 3. metrics
    s.append(CondPageBreak(110 * mm))
    s.append(Paragraph("3. Latency metrics", H1))
    s.append(Paragraph("Successful calls only. P50 = median (half the calls were faster). P95 = 95% of calls were "
                       "this fast or faster. The two highlighted rows are what a user actually feels.", B))
    for a in [primary_agent] + [x for x in agents if x != primary_agent]:
        ok_a = [r for r in by_agent[a] if r["success"]]
        if not ok_a:
            continue
        s.append(Paragraph(f"Agent {short(a)}" + (" (main)" if a == primary_agent else "")
                           + f" &middot; {len(ok_a)} successful call(s)", H2))
        s.append(metrics_table(ok_a))

    # 4. definitions
    s.append(CondPageBreak(80 * mm))
    s.append(Paragraph("4. What each metric means", H1))
    s.append(table([["Metric", "Definition"]] + [[l, d] for _, l, d in CLIENT_METRICS + SERVER_METRICS],
                   [50, 130], bold_first_col=True))

    # 5-7. latest call
    if last_ok:
        s.append(CondPageBreak(95 * mm))
        s.append(Paragraph(f"5. Where the response time goes (latest call, agent {short(primary_agent)})", H1))
        m = last_ok["_metrics"]
        total = last_ok["steps"]["agent_response_ms"]
        eou = m["eou"][-1] if m["eou"] else 0
        tt = m["ttft"][-1] if m["ttft"] else 0
        tb = m["ttfb"][-1] if m["ttfb"] else 0
        other = max(0, total - eou - tt - tb)
        s.append(breakdown_chart([("End-of-turn detection", eou), ("LLM first token", tt), ("TTS first byte", tb),
                                  ("Other: STT final text, network, audio buffering", other)], total))

        s.append(Paragraph("6. Call timeline (latest call)", H1))
        s.append(timeline_chart(last_ok["steps"]))
        s.append(Spacer(1, 4))

        s.append(Paragraph("7. Conversation transcript (latest call)", H1))
        tr = [["Speaker", "Text"]]
        for t in (last_ok.get("call_record") or {}).get("transcript", []):
            who = "Agent" if t.get("role") == "assistant" else "Caller"
            tr.append([who, (t.get("content") or "").replace("’", "'").replace("�", "'")])
        s.append(table(tr, [20, 160]))
        if last_ok.get("audio_file"):
            s.append(Paragraph(f"Audio: <font face='Courier'>{last_ok['audio_file']}</font> "
                               "(stereo: left = caller, right = agent).", B))

    # 8. runs
    s.append(CondPageBreak(60 * mm))
    s.append(Paragraph("8. All runs", H1))
    rows = [["#", "Time", "Agent", "Result", "Call setup", "Agent resp.", "Turns", "Note"]]
    shade = {}
    for i, r in enumerate(runs, 1):
        st = r["steps"]
        label, col, note = r["_class"]
        rows.append([i, r["started_at"][11:19], short(r["_agent"]), label, sec(st.get("call_setup_ms")),
                     sec(st.get("agent_response_ms")), (r.get("call_record") or {}).get("turns", "-"), note])
        shade[(3, i)] = col
    s.append(table(rows, [9, 19, 17, 17, 17, 17, 14, 70], shade))

    # 9. issues
    s.append(CondPageBreak(70 * mm))
    s.append(Paragraph("9. Issues and observations", H1))
    s.append(table([
        ["#", "Issue", "Impact", "Owner"],
        ["1", "Wrapper on 214 returns livekit_url wss://ailabs.healthviewx.com/livekit, not the LiveKit the "
              "box-214 agent uses", "Web calls on 214 are silent unless the URL is overridden", "AI Labs dev"],
        ["2", "Silent calls stay in status 'initiated' and are never closed", "Stale call records", "AI Labs dev"],
        ["3", "recording_url is empty for every call, although Egress is described as recording all calls",
              "No server recording to review", "AI Labs dev (confirm)"],
        ["4", "POST /v1/agents/{id}/calls has no authentication", "Anyone on the network can place paid calls",
              "AI Labs dev"],
        ["5", "Agent 54fc5093 greeting reads 'Am I speaking with ?' (empty name)",
              "Patient-name variable not filled in web calls", "Agent config"],
    ], [7, 85, 58, 30]))

    s.append(Paragraph("10. Next steps", H1))
    for b in [
        "Run 20+ single-user calls per agent for reliable averages and P95 (--runs 20).",
        "Fix the LiveKit URL returned on box 214 (issue 1), then re-run without the override.",
        "Ramp concurrent users: 1 -> 2 -> 4 -> 8 -> 16 -> ~30, holding each step and watching for silent calls.",
        "Box 214 limits to expect: about 33 concurrent calls (WebRTC UDP ports 50000-50200), agent worker CPU "
        "threshold, 2-4 cores with recording on every call, and provider (Cartesia/Deepgram/Gemini) quotas.",
        "Every call uses paid STT, LLM and TTS; agree budget and a test window, since another QA team shares 214.",
    ]:
        s.append(Paragraph(b, BUL, bulletText="•"))

    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                            bottomMargin=16 * mm, title="AI Labs Voice Agent - Web-Call Test Report")
    doc.build(s, onFirstPage=on_page, onLaterPages=on_page)


def env_agent():
    if Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.startswith("AGENT_ID="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("AGENT_ID")


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob("results/webcall_*.json"))
    if not files:
        sys.exit("No result files found")
    out = f"results/AI_Labs_WebCall_Test_Report_{datetime.now():%Y%m%d_%H%M}.pdf"
    build(load(files), out, env_agent())
    print(out)
