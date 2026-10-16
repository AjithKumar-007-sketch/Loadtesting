"""Load generator report: what the test bot measures by itself, without the AI Labs call-record API.

Reads only the bot side of webcall_load_test.py result files (steps, turns, events); call_record is ignored.
Sections: run summary, per call, agent reply metrics, slowest 10%, every turn (caller said / agent replied),
and what needs the application metrics API.

Usage:
    python loadgen_report.py --file results\\dynamic_patient_*_20261013_1337*.json
    python loadgen_report.py --file a.json b.json --pdf my_report.pdf --target 2500
"""
import argparse
import glob
import json
import os
import statistics
import sys
from datetime import datetime, timedelta
from xml.sax.saxutils import escape

OVERLAP_MS = 1000  # a reply this soon after the caller stops = agent spoke over / before the caller finished


def pct(v, p):
    """Percentile with linear interpolation between closest ranks (same as Excel PERCENTILE)."""
    s = sorted(v)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load_call(path):
    d = json.load(open(path, encoding="utf-8"))
    st = d.get("steps") or {}
    ev = d.get("events") or []
    at = d.get("turns") or []
    for k, t in enumerate(at):
        # the agent's words after caller turn N, and any hold during them, are recorded on caller turn N+1
        nxt = at[k + 1] if k + 1 < len(at) else {}
        if t.get("agent_response_ms") is None:
            t["_reply_text"] = "(no reply)"
        else:
            t["_reply_text"] = nxt.get("agent_text") or "(last reply - not captured by the bot)"
        t["_hold"] = nxt.get("agent_hold_ms")
    turns = [t for t in at if t.get("agent_response_ms") is not None]
    errs = [t["error"] for t in at if t.get("error")]
    p = d.get("patient") or {}
    return {
        "file": os.path.basename(path), "cid": d.get("call_id") or os.path.basename(path),
        "patient": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or "-",
        "started": d.get("started_at"), "server": d.get("livekit_url_used") or "",
        "web": st.get("web_call_ms"), "join": st.get("room_join_ms"), "greet": st.get("greeting_start_ms"),
        "total_ms": st.get("total_ms") or 0,
        "resp": [t["agent_response_ms"] for t in turns],
        "reply": [t.get("reply_duration_ms") or 0 for t in turns],
        "gap": [t["caller_timing"]["caller_gap_ms"] for t in turns if t.get("caller_timing")],
        "turns": turns, "all_turns": at, "error": d.get("error") or (errs[0] if errs else ""),
        "holds": [t["agent_hold_ms"] for t in at if t.get("agent_hold_ms")],
        "text_sources": {t.get("agent_text_source") for t in at if t.get("agent_text_source")},
        "ended": ("Caller said goodbye" if any(e["event"] in ("call_completed", "call_ended_cleanly") for e in ev)
                  else "Agent/room disconnected" if any(e["event"] == "disconnected" for e in ev)
                  else "Stopped: no agent reply" if errs or d.get("error") else "Ended"),
    }


def max_concurrent(calls):
    """Most calls that were running at the same moment, from each call's start time and length."""
    spans = []
    for c in calls:
        try:
            start = datetime.fromisoformat(c["started"])
        except (TypeError, ValueError):
            continue
        spans += [(start, 1), (start + timedelta(milliseconds=c["total_ms"]), -1)]
    best = cur = 0
    for _, step in sorted(spans, key=lambda x: (x[0], x[1])):
        cur += step
        best = max(best, cur)
    return best or len(calls)


def write_pdf(path, calls, target):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    small = ss["BodyText"].clone("small", fontSize=8, leading=10)
    hstyle = ss["BodyText"].clone("hdr", fontSize=8, leading=9, textColor=colors.white, fontName="Helvetica-Bold")
    red, hdr = colors.HexColor("#c0392b"), colors.HexColor("#2c3e50")

    def grid(data, widths, red_rows=()):
        data = [[Paragraph(h, hstyle) for h in data[0]]] + data[1:]
        t = Table(data, colWidths=widths, repeatRows=1)
        style = [("BACKGROUND", (0, 0), (-1, 0), hdr), ("FONTSIZE", (0, 0), (-1, -1), 8),
                 ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bdc3c7")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")])]
        for r in red_rows:
            style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fdecea")),
                      ("TEXTCOLOR", (-1, r), (-1, r), red), ("FONTNAME", (-1, r), (-1, r), "Helvetica-Bold")]
        t.setStyle(TableStyle(style))
        return t

    def stats(name, v):
        return ([name, f"{pct(v, 90):,.0f}", f"{statistics.median(v):,.0f}", f"{statistics.mean(v):,.0f}",
                 f"{min(v):,.0f}", f"{max(v):,.0f}"] if v else [name] + ["n/a"] * 5)

    ms = lambda v: f"{v:,.0f}" if v is not None else "-"  # noqa: E731
    all_resp = [v for c in calls for v in c["resp"]]
    run_p90 = pct(all_resp, 90) if all_resp else None
    starts = sorted(c["started"] for c in calls if c["started"])
    when = datetime.fromisoformat(starts[0]).strftime("%d %b %Y %H:%M") if starts else "unknown time"
    servers = sorted({c["server"] for c in calls if c["server"]})
    conc = max_concurrent(calls)

    story = [Paragraph("Load generator metrics - without application metrics API", ss["Title"]),
             Paragraph(escape(f"Built only from what the test bot measures itself (webcall_load_test.py: steps, "
                              f"turns, events). The AI Labs call-record API (GET /v1/calls/<id>) is NOT used. "
                              f"Run: {len(calls)} call(s), up to {conc} at once, started {when}, "
                              f"LiveKit {', '.join(servers) or '-'}. Target {target:,} ms. "
                              f"All times in ms, bot clock. Generated {datetime.now():%Y-%m-%d %H:%M}."), small),
             Spacer(1, 4 * mm)]

    # 1. run summary
    over = sum(1 for v in all_resp if v > target)
    fail_calls = sum(1 for c in calls if c["resp"] and pct(c["resp"], 90) > target)
    verdict = "n/a" if run_p90 is None else ("FAIL" if run_p90 > target else "PASS")
    story += [Paragraph("1. Run summary", ss["Heading2"]),
              grid([["Metric", "P90", "Median", "Average", "Min", "Max"],
                    stats("Agent response time (caller stops -> agent voice heard)", all_resp),
                    stats("Time to greeting (join -> first agent audio)", [c["greet"] for c in calls if c["greet"]]),
                    stats("Web-call API create time", [c["web"] for c in calls if c["web"]]),
                    stats("Room join time", [c["join"] for c in calls if c["join"]]),
                    stats("Agent reply length (agent speaking time)", [v for c in calls for v in c["reply"]]),
                    stats("Caller bot gap (agent stops -> caller speaks)", [v for c in calls for v in c["gap"]])],
                   [95 * mm, 26 * mm, 26 * mm, 26 * mm, 26 * mm, 26 * mm]),
              Spacer(1, 3 * mm),
              Paragraph(f"<b>Calls:</b> {len(calls)} started, {sum(1 for c in calls if c['resp'])} connected and "
                        f"answered, {sum(1 for c in calls if c['error'])} with an error. <b>Turns:</b> "
                        f"{len(all_resp)} answered; <font color='#c0392b'><b>{over} of {len(all_resp)} over "
                        f"{target:,} ms</b></font>. <b>Agent response P90 {ms(run_p90)} ms</b> ({verdict} vs "
                        f"{target:,} ms); {fail_calls} of {len(calls)} calls have their own P90 above target.",
                        ss["BodyText"])]

    # 2. per call
    rows = [["Call", "Patient", "Web-call API", "Room join", "Time to greeting", "Turns answered",
             "Response median", "Response P90", "Slowest", f"Turns > {target / 1000:g} s", "Call length (s)", "Result"]]
    red_rows = []
    for n, c in enumerate(calls, 1):
        p90 = pct(c["resp"], 90) if c["resp"] else None
        rows.append([c["cid"][:12] + "...", Paragraph(escape(c["patient"]), small), ms(c["web"]), ms(c["join"]),
                     ms(c["greet"]), len(c["resp"]), ms(statistics.median(c["resp"]) if c["resp"] else None),
                     ms(p90), ms(max(c["resp"]) if c["resp"] else None), sum(v > target for v in c["resp"]),
                     f"{c['total_ms'] / 1000:.0f}",
                     Paragraph(escape(("ERROR: " + c["error"]) if c["error"] else "Completed"), small)])
        if (p90 or 0) > target or c["error"]:
            red_rows.append(n)
    story += [Spacer(1, 4 * mm), Paragraph("2. Per call (red = P90 above target or call error)", ss["Heading2"]),
              grid(rows, [27 * mm, 25 * mm, 19 * mm, 16 * mm, 21 * mm, 18 * mm, 20 * mm, 19 * mm, 16 * mm, 17 * mm,
                          17 * mm, 58 * mm], red_rows)]

    # 3. agent reply metrics
    rows = [["Call", "Caller turns sent", "Agent replied", "No reply", "Reply rate", "Response median",
             "Response P90", "Reply length median", "Holds > 1 s (tool running)", "Longest hold",
             "Spoke over caller (< 1 s)", "How the call ended"]]
    red_rows = []
    tot = {"sent": 0, "rep": 0, "no": 0, "holds": [], "over": 0}
    for n, c in enumerate(calls, 1):
        sent, rep_n = len(c["all_turns"]), len(c["resp"])
        talk_over = sum(1 for v in c["resp"] if v < OVERLAP_MS)
        tot["sent"] += sent
        tot["rep"] += rep_n
        tot["no"] += sent - rep_n
        tot["holds"] += c["holds"]
        tot["over"] += talk_over
        rows.append([c["cid"][:12] + "...", sent, rep_n, sent - rep_n, f"{rep_n / sent:.0%}" if sent else "-",
                     ms(statistics.median(c["resp"]) if c["resp"] else None),
                     ms(pct(c["resp"], 90) if c["resp"] else None),
                     ms(statistics.median(c["reply"]) if c["reply"] else None),
                     len(c["holds"]), ms(max(c["holds"]) if c["holds"] else None), talk_over, c["ended"]])
        if sent - rep_n:
            red_rows.append(n)
    all_reply = [v for c in calls for v in c["reply"]]
    rows.append(["All calls", tot["sent"], tot["rep"], tot["no"],
                 f"{tot['rep'] / tot['sent']:.0%}" if tot["sent"] else "-",
                 ms(statistics.median(all_resp) if all_resp else None), ms(run_p90),
                 ms(statistics.median(all_reply) if all_reply else None), len(tot["holds"]),
                 ms(max(tot["holds"]) if tot["holds"] else None), tot["over"], ""])
    if tot["no"]:
        red_rows.append(len(rows) - 1)
    rate = f"{tot['rep'] / tot['sent']:.0%}" if tot["sent"] else "-"
    story += [PageBreak(), Paragraph("3. Agent reply metrics (red = agent missed a turn)", ss["Heading2"]),
              grid(rows, [27 * mm, 18 * mm, 17 * mm, 15 * mm, 15 * mm, 20 * mm, 18 * mm, 21 * mm, 24 * mm, 18 * mm,
                          22 * mm, 58 * mm], red_rows),
              Spacer(1, 2 * mm),
              Paragraph(f"Agent replied to <b>{tot['rep']} of {tot['sent']} caller turns ({rate})</b>; {tot['no']} "
                        f"turn(s) got no reply. Hold = the agent paused mid-reply (e.g. 'let me check') while a tool "
                        f"ran; {len(tot['holds'])} hold(s), longest "
                        f"{(max(tot['holds']) / 1000 if tot['holds'] else 0):.1f} s. 'Spoke over caller' = reply "
                        f"started under {OVERLAP_MS} ms after the caller stopped, so the agent likely talked before "
                        f"the caller finished.", small)]

    # 4. slowest 10%
    if all_resp:
        tail = sorted([(c["cid"], t) for c in calls for t in c["turns"] if t["agent_response_ms"] > run_p90],
                      key=lambda x: -x[1]["agent_response_ms"])
        rows = [["Call", "Turn", "Caller said", "Agent replied", "Agent response", "Agent reply length"]]
        for cid, t in tail:
            rows.append([cid[:12] + "...", t["turn"],
                         Paragraph(escape(t.get("caller_spoken") or t.get("text") or ""), small),
                         Paragraph(escape(t["_reply_text"]), small),
                         ms(t["agent_response_ms"]), ms(t.get("reply_duration_ms") or 0)])
        story += [PageBreak(), Paragraph(f"4. Slowest 10% - turns above run P90 ({run_p90:,.0f} ms)", ss["Heading2"]),
                  grid(rows, [27 * mm, 11 * mm, 60 * mm, 120 * mm, 24 * mm, 28 * mm])]

    # 5. every turn
    story += [PageBreak(), Paragraph("5. Every turn - caller said / agent replied (red = no reply or over target)",
                                     ss["Heading2"])]
    rows = [["Call", "Turn", "Caller said", "Agent replied", "Agent response", "Agent reply length", "Hold"]]
    red_rows = []
    for c in calls:
        for t in c["all_turns"]:
            replied = t.get("agent_response_ms") is not None
            rows.append([c["cid"][:12] + "...", t.get("turn"),
                         Paragraph(escape(t.get("caller_spoken") or t.get("text") or ""), small),
                         Paragraph(escape(t["_reply_text"]), small),
                         ms(t["agent_response_ms"]) if replied else "NO REPLY",
                         ms(t.get("reply_duration_ms") or 0) if replied else "-", ms(t.get("_hold"))])
            if not replied or t["agent_response_ms"] > target:
                red_rows.append(len(rows) - 1)
    sources = set().union(*(c["text_sources"] for c in calls)) if calls else set()
    src = ("read live from the AI Labs transcript" if sources == {"transcript"}
           else "transcribed from the agent's audio" if sources == {"audio"}
           else "from the AI Labs transcript or the agent's audio" if sources else "not recorded")
    story += [grid(rows, [26 * mm, 11 * mm, 57 * mm, 116 * mm, 22 * mm, 22 * mm, 16 * mm], red_rows),
              Paragraph(f"Agent replied = the agent's words after that caller turn ({src} by the bot). The agent's "
                        f"final goodbye is not captured by the bot.", small)]

    # 6. what is / is not available
    story += [PageBreak(), Paragraph("6. What the load generator gives - and what needs the application metrics API",
                                     ss["Heading2"]),
              grid([["Metric", "Load generator only", "Needs GET /v1/calls API"],
                    ["Web-call create time, room join time, time to greeting", "Yes", "-"],
                    ["Agent response time per turn (caller stops -> agent voice heard), P90 / median", "Yes", "-"],
                    ["Agent reply length, caller speaking time, call length", "Yes", "-"],
                    ["Errors: no reply, disconnects, silent call, connection failure", "Yes", "-"],
                    ["Calls per run / concurrency, pass-fail vs target", "Yes", "-"],
                    ["Agent replied or not per turn, reply rate, holds, talking over the caller", "Yes", "-"],
                    ["STT time (caller stops -> final transcript)", "Yes, if the agent publishes room transcriptions",
                     "-"],
                    ["What the agent said (text)", "Only by speech-to-text of the agent audio (Whisper)",
                     "Yes (transcript)"],
                    ["Booking result / appointment ID", "Only if parsed from the agent's words", "Yes"],
                    ["Split of the wait: EOU / LLM TTFT / TTS / other", "No", "Yes (flow_trace metrics)"],
                    ["LLM tokens, number of model calls, tool / API calls", "No", "Yes"],
                    ["Server-clock real wait (no network in between)", "No - bot clock includes network both ways",
                     "Yes"]],
                   [120 * mm, 75 * mm, 60 * mm]),
              Spacer(1, 4 * mm),
              Paragraph("Note: the bot starts the response timer when its audio clip has been sent, including the "
                        "silence at the end of the clip, so bot-measured response can read slightly lower than the "
                        "server's real wait; it also includes network time both ways.", small)]

    SimpleDocTemplate(path, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm, topMargin=12 * mm,
                      bottomMargin=12 * mm, title="Load generator metrics").build(story)
    return run_p90


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", action="extend", nargs="+", required=True,
                    help="webcall_load_test.py result JSON(s); wildcards allowed")
    ap.add_argument("--pdf", help="output PDF (default: loadgen_report_<timestamp>.pdf)")
    ap.add_argument("--target", type=int, default=int(os.environ.get("LK_TARGET_MS", "2500")),
                    help="agent response target in ms (default 2500, or LK_TARGET_MS)")
    opts = ap.parse_args()

    # Windows shells do not expand wildcards, so expand them here
    files = [m for f in opts.file for m in (sorted(glob.glob(f)) or [f])]
    calls = []
    for f in files:
        try:
            calls.append(load_call(f))
        except (OSError, ValueError, KeyError) as e:
            print(f"skipped {f}: {e}")
    if not calls:
        sys.exit("No result files could be read.")
    calls.sort(key=lambda c: c["started"] or "")

    path = opts.pdf or f"loadgen_report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    try:
        run_p90 = write_pdf(path, calls, opts.target)
    except ImportError:
        sys.exit("PDF needs reportlab: pip install reportlab")
    all_resp = [v for c in calls for v in c["resp"]]
    sent = sum(len(c["all_turns"]) for c in calls)
    print(f"{len(calls)} call(s), {len(all_resp)} of {sent} caller turns answered")
    if all_resp:
        print(f"agent response: median {statistics.median(all_resp):,.0f} ms | P90 {run_p90:,.0f} ms | "
              f"{sum(v > opts.target for v in all_resp)} over {opts.target:,} ms -> "
              f"{'FAIL' if run_p90 > opts.target else 'PASS'}")
    print(f"PDF: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
