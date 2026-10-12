#!/usr/bin/env python3
"""
Builds a PDF report for dynamic (AI caller, multi-turn) runs of webcall_load_test.py.

Usage (needs reportlab):
    python make_dynamic_report.py results/dynamic_patient_1_....json [more.json ...]
    python make_dynamic_report.py            # latest dynamic run
"""

import glob
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import CondPageBreak, Paragraph, SimpleDocTemplate, Spacer

from make_report import (B, BUL, H1, H2, NOTE, SEG, SUB, TITLE, GRID, NAVY, OK, BAD, WARN,
                         kpis, sec, server_metrics, stats, table)

AGENT_COL, CALLER_COL = SEG[0], SEG[2]


def clean(s):
    return (s or "").replace("—", "-").replace("’", "'").replace("�", "'").replace("–", "-")


def turn_chart(turns):
    """Grouped bars per turn: agent response latency vs caller (test-side) delay."""
    W, H = 180 * mm, 62 * mm
    d = Drawing(W, H)
    x0, y0, ch = 30, 22, H - 44
    vals = [(t.get("agent_response_ms"), (t.get("caller_timing") or {}).get("caller_gap_ms")) for t in turns]
    top = max([v for pair in vals for v in pair if v] + [1000])
    top = ((int(top / 1000) // 2) + 1) * 2000
    bw_group = (W - x0 - 10) / max(len(vals), 1)
    bw = min(16, bw_group / 2.6)
    for s_ in range(0, int(top / 1000) + 1, 2):
        y = y0 + ch * s_ * 1000 / top
        d.add(Line(x0, y, W - 10, y, strokeColor=colors.HexColor("#e3e7ee"), strokeWidth=0.5))
        d.add(String(x0 - 4, y - 3, f"{s_}s", textAnchor="end", fontName="Helvetica", fontSize=7,
                     fillColor=colors.HexColor("#666")))
    for i, (a, c) in enumerate(vals):
        gx = x0 + i * bw_group + (bw_group - 2 * bw - 3) / 2
        for j, (v, col) in enumerate([(a, AGENT_COL), (c, CALLER_COL)]):
            if v:
                h = ch * v / top
                d.add(Rect(gx + j * (bw + 3), y0, bw, h, fillColor=col, strokeColor=col))
        d.add(String(gx + bw + 1.5, y0 - 10, f"T{i + 1}", textAnchor="middle", fontName="Helvetica", fontSize=7.5,
                     fillColor=colors.HexColor("#333")))
    for k, (lab, col) in enumerate([("Agent response latency (AI Labs)", AGENT_COL),
                                    ("Test caller delay (our side)", CALLER_COL)]):
        lx = x0 + k * 190
        d.add(Rect(lx, H - 12, 8, 8, fillColor=col, strokeColor=col))
        d.add(String(lx + 12, H - 11, lab, fontName="Helvetica", fontSize=8, fillColor=colors.HexColor("#333")))
    return d


def to_local(iso_utc, offset):
    return (datetime.fromisoformat(iso_utc).replace(tzinfo=None) + offset).strftime("%H:%M:%S")


def run_section(r, idx):
    s = []
    st = r["steps"]
    turns = r.get("turns", [])
    rec = r.get("call_record") or {}
    pat = r.get("patient") or {}
    an = rec.get("post_call_analysis") or {}
    agent_ms = [t["agent_response_ms"] for t in turns if t.get("agent_response_ms")]
    gaps = [t["caller_timing"]["caller_gap_ms"] for t in turns if t.get("caller_timing")]
    sa, sg = stats(agent_ms), stats(gaps)
    m = server_metrics(rec)

    name = f"{pat.get('first_name', '')} {pat.get('last_name', '')}".strip() or f"run {idx}"
    s.append(Paragraph(f"{idx}. Call: {name}", H1))
    ok = r.get("success")
    goal = an.get("call_successful")
    s.append(Paragraph(
        f"<b>Test result: {'PASS' if ok else 'FAIL'}</b>"
        + (f" &middot; <b>Agent outcome: {'goal reached' if goal else 'goal not reached'}</b>" if goal is not None else "")
        + f" &middot; {len(turns)} caller turns &middot; started {r['started_at'][:19].replace('T', ' ')}"
        + (f" &middot; <font color='#a12a2a'>{r.get('error')}</font>" if r.get("error") else ""), B))
    s.append(kpis([
        (sec(st.get("call_setup_ms")), "call setup latency<br/>(start -> greeting heard)"),
        (sec(sa["avg"]) if sa else "-", "avg agent response<br/>latency"),
        (sec(sa["max"]) if sa else "-", "slowest agent<br/>response"),
        (sec(sg["avg"]) if sg else "-", "avg test-caller<br/>delay (our side)"),
        (sec(st.get("total_ms")), "total call<br/>duration"),
    ]))
    s.append(Spacer(1, 4))

    s.append(Paragraph("Caller persona", H2))
    s.append(table([["Name", "Date of birth", "Phone", "Specialty", "Caller LLM"],
                    [name, pat.get("dob", "-"), pat.get("phone", "-"), pat.get("specialty", "-"),
                     next((t.get("llm_model") for t in turns if t.get("llm_model")), "-")]],
                   [36, 34, 30, 30, 50]))

    s.append(Paragraph("Latency per turn", H2))
    s.append(turn_chart(turns))
    rows = [["#", "Caller said", "Agent response", "Agent reply length", "Caller delay", "= wait + LLM + speech",
             "Input"]]
    shade = {}
    for i, t in enumerate(turns, 1):
        c = t.get("caller_timing") or {}
        rows.append([t["turn"], clean(t.get("caller_spoken")), sec(t.get("agent_response_ms")),
                     sec(t.get("reply_duration_ms")), sec(c.get("caller_gap_ms")),
                     f"{sec(c.get('detect_wait_ms'))} + {sec(c.get('llm_ms'))} + {sec(c.get('tts_ms'))}" if c else "-",
                     t.get("agent_text_source", "-")])
        a = t.get("agent_response_ms")
        if a:
            shade[(2, i)] = OK if a < 3000 else WARN if a < 6000 else BAD
    s.append(table(rows, [8, 47, 19, 19, 17, 48, 22], shade))
    s.append(Paragraph("Agent response colour: green under 3 s, amber 3-6 s, red over 6 s. "
                       "<b>Caller delay</b> is the test tool's own time to answer (not AI Labs): wait = making sure "
                       "the agent finished, LLM = test caller thinking, speech = generating its voice.", NOTE))

    s.append(CondPageBreak(60 * mm))
    s.append(Paragraph("AI Labs agent, server-side timings (from the call record)", H2))
    mrows = [["Stage", "n", "Avg", "Min", "P50", "P95", "Max"]]
    for key, label in [("eou", "End-of-turn delay"), ("ttft", "LLM time to first token"),
                       ("ttfb", "TTS time to first byte")]:
        x = stats(m[key])
        if x:
            mrows.append([label, x["n"], sec(x["avg"]), sec(x["min"]), sec(x["p50"]), sec(x["p95"]), sec(x["max"])])
    for label, x in [("Agent response (caller-measured)", sa), ("Test-caller delay (our side)", sg)]:
        if x:
            mrows.append([label, x["n"], sec(x["avg"]), sec(x["min"]), sec(x["p50"]), sec(x["p95"]), sec(x["max"])])
    s.append(table(mrows, [58, 12, 22, 22, 22, 22, 22]))

    if an:
        s.append(Paragraph("Post-call analysis (AI Labs)", H2))
        s.append(table([["Goal reached", "Sentiment", "Summary"],
                        ["Yes" if an.get("call_successful") else "No", clean(an.get("user_sentiment")),
                         clean(an.get("call_summary"))]], [24, 32, 124],
                       {(0, 1): OK if an.get("call_successful") else BAD}))

    tr = rec.get("transcript", [])
    if tr:
        s.append(CondPageBreak(50 * mm))
        s.append(Paragraph("Conversation transcript", H2))
        offset = timedelta(hours=5, minutes=30)  # call record times are UTC; report in IST
        rows = [["Time", "Speaker", "Text"]]
        for x in tr:
            rows.append([to_local(x["at"], offset) if x.get("at") else "",
                         "Agent" if x.get("role") == "assistant" else "Caller", clean(x.get("content"))])
        s.append(table(rows, [18, 17, 145]))
    if r.get("audio_file"):
        s.append(Paragraph(f"Recording: <font face='Courier'>{r['audio_file']}</font> "
                           "(stereo: left = test caller, right = agent).", B))

    obs = observations(r, turns, sa, sg)
    if obs:
        s.append(Paragraph("Observations", H2))
        s += [Paragraph(o, BUL, bulletText="•") for o in obs]
    return s


def observations(r, turns, sa, sg):
    o = []
    slow = [t for t in turns if (t.get("agent_response_ms") or 0) >= 6000]
    if slow:
        o.append("Slow agent replies on turn(s) " + ", ".join(
            f"{t['turn']} ({sec(t['agent_response_ms'])}, after \"{clean(t.get('caller_spoken'))[:40]}\")" for t in slow)
            + ". Check the agent's LLM time and tool calls for these steps.")
    long_ = [t for t in turns if (t.get("reply_duration_ms") or 0) >= 10000]
    if long_:
        o.append("Long agent answers on turn(s) " + ", ".join(f"{t['turn']} ({sec(t['reply_duration_ms'])})"
                                                              for t in long_) + "; consider shorter spoken prompts.")
    waits = [t["caller_timing"]["detect_wait_ms"] for t in turns if t.get("caller_timing")]
    if waits and min(waits) > 2000:
        o.append(f"The test caller waited about {sec(sum(waits) / len(waits))} per turn to be sure the agent had "
                 "finished. Run with --settle 0.8 --end-of-speech 0.8 to cut this to about 1.6 s.")
    src = {t.get("agent_text_source") for t in turns}
    if src == {"transcript"}:
        o.append("The test caller used the agent's transcript text every turn (fast path); no audio fallback.")
    elif "audio" in src:
        o.append("Some turns fell back to sending audio to the caller LLM (agent text not yet stored); those turns are slower.")
    if turns and not turns[-1].get("agent_response_ms"):
        o.append(f"No agent reply after the last caller turn (\"{clean(turns[-1].get('caller_spoken'))}\"): the call "
                 "ended normally.")
    return o


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777"))
    canvas.drawString(15 * mm, 9 * mm, "AI Labs Voice Agent - Dynamic Conversation Test Report")
    canvas.drawRightString(195 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build(paths, out):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    agent = (runs[0].get("call_record") or {}).get("agent_id") or (runs[0].get("web_call_response") or {}).get("agent_id")
    s = [Paragraph("AI Labs Voice Agent - Dynamic Conversation Test", TITLE),
         Paragraph(f"AI test caller (multi-turn) &middot; agent {agent} &middot; "
                   f"generated {datetime.now():%Y-%m-%d %H:%M}", SUB), Spacer(1, 6)]
    if len(runs) > 1:
        s.append(Paragraph("Summary", H1))
        rows = [["#", "Caller", "Result", "Goal", "Turns", "Avg agent resp.", "Max agent resp.", "Avg caller delay"]]
        shade = {}
        for i, r in enumerate(runs, 1):
            t = r.get("turns", [])
            a = [x["agent_response_ms"] for x in t if x.get("agent_response_ms")]
            pat = r.get("patient") or {}
            an = (r.get("call_record") or {}).get("post_call_analysis") or {}
            rows.append([i, f"{pat.get('first_name', '')} {pat.get('last_name', '')}", "Pass" if r["success"] else "Fail",
                         {True: "Yes", False: "No"}.get(an.get("call_successful"), "-"), len(t),
                         sec(sum(a) / len(a)) if a else "-", sec(max(a)) if a else "-",
                         sec(r["steps"].get("avg_caller_gap_ms"))])
            shade[(2, i)] = OK if r["success"] else BAD
        s.append(table(rows, [8, 36, 16, 14, 14, 30, 30, 32], shade))
    s.append(Paragraph("Setup: wrapper POST /v1/agents/{id}/web-calls -> join LiveKit room as caller -> an AI test "
                       "caller (Gemini) reads each agent message and answers in character as the patient, spoken with "
                       "Windows TTS -> per-turn latency measured -> GET /v1/calls/{id} for transcript and agent "
                       "metrics.", B))
    for i, r in enumerate(runs, 1):
        s += run_section(r, i)
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                            bottomMargin=16 * mm, title="AI Labs Voice Agent - Dynamic Conversation Test")
    doc.build(s, onFirstPage=on_page, onLaterPages=on_page)


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob("results/dynamic_*.json"))[-1:]
    if not files:
        sys.exit("No dynamic result files found")
    stem = Path(files[0]).stem if len(files) == 1 else f"dynamic_{len(files)}runs"
    out = f"results/{stem}_report.pdf"
    build(files, out)
    print(out)
