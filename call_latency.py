"""Per-turn response latency (caller stops -> agent starts) for voice-agent calls.

Read only: one POST to log in, then GET /v1/calls/<id> per call. Nothing is modified.

Usage:
    python call_latency.py <call_id> [<call_id> ...]
  (asks for email/password, or reads LK_EMAIL / LK_PASSWORD; set LK_ORG_ID if the account is in several orgs)
  or analyse saved responses offline:
    python call_latency.py --file call1.json --file call2.json

Latency per caller turn = eou_delay_ms
                        + sum(llm_ttft_ms of every LLMMetrics up to the first TTSMetrics)
                        + tts_ttfb_ms of that first TTSMetrics.
"""
import argparse
import getpass
import json
import os
import statistics
import sys
from datetime import datetime
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = os.environ.get("LK_BASE_URL", "http://172.17.20.214:6002")
SLOW_MS = 3000


def http(method, path, token=None, data=None, org_id=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if org_id:
        headers["X-Org-Id"] = str(org_id)
    body = None
    if data is not None:
        body = urllib.parse.urlencode(data).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(BASE_URL + path, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def login():
    email = os.environ.get("LK_EMAIL") or input("AI Labs login email: ").strip()
    password = os.environ.get("LK_PASSWORD") or getpass.getpass("Password (hidden): ")
    try:
        return http("POST", "/v1/auth/login", data={"username": email, "password": password})["access_token"]
    except urllib.error.HTTPError as e:
        if e.code in (400, 401, 403):
            sys.exit(f"Login failed ({e.code}) for {email}: wrong email or password for {BASE_URL}.")
        raise


def fetch_call(token, call_id, org_id):
    try:
        return http("GET", f"/v1/calls/{call_id}", token=token)
    except urllib.error.HTTPError as e:
        if e.code == 404 and org_id:
            return http("GET", f"/v1/calls/{call_id}", token=token, org_id=org_id)
        if e.code == 404:
            raise RuntimeError("404 Call not found (set LK_ORG_ID if the account is in more than one org)")
        raise


def num(args, key):
    """Find a metric value in arguments, or one level down (e.g. arguments.metrics.<key>)."""
    if key in args:
        return float(args[key] or 0)
    for v in args.values():
        if isinstance(v, dict) and key in v:
            return float(v[key] or 0)
    raise KeyError(key)


def transcription_ms(args):
    """Transcription delay (caller stops -> final transcript) from an EOUMetrics item, if the engine reports it.
    LiveKit calls it transcription_delay (seconds); the API may expose it as *_ms."""
    for key in ("transcription_delay_ms", "stt_delay_ms"):
        try:
            return num(args, key)
        except KeyError:
            pass
    try:
        return num(args, "transcription_delay") * 1000
    except KeyError:
        return None


def summary(values):
    """P90 / Median / Min, like the Retell call-detail latency popup."""
    if not values:
        return "n/a"
    return f"P90 {pct(values, 90):>6.0f} ms | Median {statistics.median(values):>6.0f} ms | Min {min(values):>6.0f} ms"


def print_latency_block(turns):
    stt = [t["stt"] for t in turns if t["stt"] is not None]
    print(f"  End to end latency    {summary([t['total'] for t in turns])}")
    print(f"  Transcription latency {summary(stt) if stt else 'n/a (no transcription delay field in EOUMetrics)'}")
    print(f"  LLM latency           {summary([sum(t['llm']) for t in turns])}")
    print(f"  TTS latency           {summary([t['tts'] for t in turns])}")


def user_texts(call):
    out = []
    for t in call.get("transcript") or []:
        if t.get("role") == "user":
            out.append(t.get("content") or t.get("text") or t.get("message") or "")
    return out


def analyse(call):
    metrics = [i for i in (call.get("flow_trace") or []) if i.get("type") == "metrics"]
    if not metrics:
        return None
    turns, unanswered, current = [], [], None
    caller_idx = 0  # every EOU is one caller utterance, so it consumes one user transcript line
    texts = user_texts(call)

    def close_unanswered(reason):
        unanswered.append({**current, "reason": reason})

    for item in metrics:
        a = item.get("arguments") or {}
        src = a.get("source")
        if src == "EOUMetrics":
            if current is not None:
                close_unanswered("next EOU before any TTS")
            current = {
                "node_id": item.get("node_id"),
                "at": a.get("at"),
                "text": texts[caller_idx] if caller_idx < len(texts) else "(no transcript line)",
                "eou": num(a, "eou_delay_ms"),
                "stt": transcription_ms(a),
                "llm": [],
            }
            caller_idx += 1
        elif src == "LLMMetrics" and current is not None:
            current["llm"].append(num(a, "llm_ttft_ms"))
        elif src == "TTSMetrics" and current is not None:
            tts = num(a, "tts_ttfb_ms")
            current["tts"] = tts
            current["total"] = current["eou"] + sum(current["llm"]) + tts
            turns.append(current)
            current = None
        # STTMetrics, and LLM/TTS with no open caller turn (greeting / follow-on speech), are ignored
    if current is not None:
        close_unanswered("call ended before any TTS")
    return {"turns": turns, "unanswered": unanswered, "user_lines": len(texts), "eou_count": caller_idx}


def pct(values, p):
    """Percentile with linear interpolation between closest ranks."""
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def biggest_part(t):
    model = sum(t["llm"])
    parts = {"wait (EOU)": t["eou"], f"model ({len(t['llm'])} call(s))": model, "voice (TTS)": t["tts"]}
    name, ms = max(parts.items(), key=lambda kv: kv[1])
    return f"{name} {ms:.0f} ms = {100 * ms / t['total']:.0f}%"


def short(s, n=45):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def stats_row(name, values):
    if not values:
        return [name, "n/a", "n/a", "n/a", "n/a"]
    return [name, f"{pct(values, 90):.0f} ms", f"{statistics.median(values):.0f} ms",
            f"{statistics.mean(values):.0f} ms", f"{min(values):.0f} ms"]


def write_pdf(path, report, all_turns):
    """PDF version of the report: summary per call, turn table, breakdown chart, slow turns."""
    from xml.sax.saxutils import escape
    try:
        import reportlab  # noqa: F401
    except ImportError:
        raise RuntimeError("PDF needs reportlab: pip install reportlab") from None
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.legends import Legend
    from reportlab.graphics.shapes import Drawing, Line, String
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    small = ss["BodyText"].clone("small", fontSize=8, leading=10)
    red, header_bg = colors.HexColor("#c0392b"), colors.HexColor("#2c3e50")
    c_eou, c_llm, c_tts = colors.HexColor("#95a5a6"), colors.HexColor("#2980b9"), colors.HexColor("#27ae60")

    def grid(data, widths, slow_rows=()):
        t = Table(data, colWidths=widths, repeatRows=1)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), header_bg), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bdc3c7")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
        ]
        for r in slow_rows:
            style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fdecea")), ("TEXTCOLOR", (-1, r), (-1, r), red),
                      ("FONTNAME", (-1, r), (-1, r), "Helvetica-Bold")]
        t.setStyle(TableStyle(style))
        return t

    def latency_table(turns):
        stt = [t["stt"] for t in turns if t["stt"] is not None]
        return grid([["Latency", "P90", "Median", "Average", "Min"],
                     stats_row("End to end", [t["total"] for t in turns]),
                     stats_row("Transcription", stt),
                     stats_row("LLM (all model calls)", [sum(t["llm"]) for t in turns]),
                     stats_row("TTS (first audio)", [t["tts"] for t in turns]),
                     stats_row("EOU wait", [t["eou"] for t in turns])],
                    [55 * mm, 30 * mm, 30 * mm, 30 * mm, 30 * mm])

    def chart(turns):
        w, h = 250 * mm, 70 * mm
        d = Drawing(w, h)
        bc = VerticalBarChart()
        bc.x, bc.y, bc.width, bc.height = 15 * mm, 12 * mm, w - 60 * mm, h - 20 * mm
        bc.data = [[t["eou"] for t in turns], [sum(t["llm"]) for t in turns], [t["tts"] for t in turns]]
        bc.categoryAxis.style = "stacked"
        bc.categoryAxis.categoryNames = [str(i) for i in range(1, len(turns) + 1)]
        bc.categoryAxis.labels.fontSize = 7
        bc.valueAxis.valueMin = 0
        top = max(max(t["total"] for t in turns), SLOW_MS) * 1.1
        bc.valueAxis.valueMax = top
        bc.valueAxis.labels.fontSize = 7
        for i, c in enumerate((c_eou, c_llm, c_tts)):
            bc.bars[i].fillColor, bc.bars[i].strokeColor = c, None
        d.add(bc)
        y3 = bc.y + bc.height * SLOW_MS / top
        d.add(Line(bc.x, y3, bc.x + bc.width, y3, strokeColor=red, strokeDashArray=[3, 2]))
        d.add(String(bc.x + bc.width + 2, y3 - 3, "3 s", fontSize=7, fillColor=red))
        lg = Legend()
        lg.x, lg.y, lg.fontSize = bc.x + bc.width + 12 * mm, h - 15 * mm, 8
        lg.colorNamePairs = [(c_eou, "EOU wait"), (c_llm, "LLM"), (c_tts, "TTS")]
        d.add(lg)
        d.add(String(bc.x, 2, "Turn #  (ms per turn)", fontSize=7))
        return d

    doc = SimpleDocTemplate(path, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title="Voice agent response latency")
    story = [Paragraph("Voice agent response latency", ss["Title"]),
             Paragraph(escape(f"Generated {datetime.now():%Y-%m-%d %H:%M} | source {BASE_URL} | "
                              f"latency = EOU wait + all model calls before reply + TTS first audio | slow > {SLOW_MS} ms"),
                       small), Spacer(1, 4 * mm)]

    if all_turns:
        totals = [t["total"] for _, _, t in all_turns]
        n_slow = sum(1 for v in totals if v > SLOW_MS)
        story += [Paragraph(f"All calls: {len(report)} call(s), {len(totals)} answered caller turns, "
                            f"<font color='#c0392b'><b>{n_slow} above 3 s</b></font>", ss["Heading2"]),
                  latency_table([t for _, _, t in all_turns]), Spacer(1, 4 * mm)]
        slow = [x for x in all_turns if x[2]["total"] > SLOW_MS]
        if slow:
            rows = [["Call", "Turn", "Node", "TOTAL", "EOU wait", "LLM", "# model calls", "TTS", "Biggest part"]]
            for cid, i, t in slow:
                rows.append([short(cid, 34), i, short(t["node_id"], 24), f"{t['total']:.0f}", f"{t['eou']:.0f}",
                             f"{sum(t['llm']):.0f}", len(t["llm"]), f"{t['tts']:.0f}", biggest_part(t)])
            story += [Paragraph("Slow turns (> 3 s)", ss["Heading3"]),
                      grid(rows, [62 * mm, 12 * mm, 42 * mm, 18 * mm, 20 * mm, 18 * mm, 22 * mm, 16 * mm, 60 * mm])]

    for cid, res, err in report:
        story += [PageBreak(), Paragraph(escape(f"Call {cid}"), ss["Heading2"])]
        if err:
            story.append(Paragraph(escape(f"Could not fetch: {err}"), ss["BodyText"]))
            continue
        if res is None:
            story.append(Paragraph("No 'metrics' items in flow_trace - skipped.", ss["BodyText"]))
            continue
        turns = res["turns"]
        if res["user_lines"] != res["eou_count"]:
            story.append(Paragraph(f"Note: {res['eou_count']} EOU items vs {res['user_lines']} user transcript lines "
                                   f"- caller text may be misaligned.", small))
        if not turns:
            story.append(Paragraph("No answered caller turns.", ss["BodyText"]))
        else:
            story += [latency_table(turns), Spacer(1, 3 * mm), chart(turns), Spacer(1, 3 * mm)]
            rows = [["#", "Node", "Caller text", "EOU ms", "# model", "LLM ms", "TTS ms", "TOTAL ms"]]
            slow_rows = []
            for i, t in enumerate(turns, 1):
                rows.append([i, Paragraph(escape(str(t["node_id"])), small), Paragraph(escape(t["text"]), small),
                             f"{t['eou']:.0f}", len(t["llm"]), f"{sum(t['llm']):.0f}", f"{t['tts']:.0f}",
                             f"{t['total']:.0f}" + (" SLOW" if t["total"] > SLOW_MS else "")])
                if t["total"] > SLOW_MS:
                    slow_rows.append(i)
            story.append(grid(rows, [10 * mm, 40 * mm, 120 * mm, 18 * mm, 16 * mm, 18 * mm, 18 * mm, 22 * mm], slow_rows))
        if res["unanswered"]:
            rows = [["Node", "Caller text", "EOU ms", "Reason"]]
            for u in res["unanswered"]:
                rows.append([Paragraph(escape(str(u["node_id"])), small), Paragraph(escape(u["text"]), small),
                             f"{u['eou']:.0f}", u["reason"]])
            story += [Paragraph("Not counted - agent did not reply to this EOU", ss["Heading3"]),
                      grid(rows, [45 * mm, 140 * mm, 20 * mm, 55 * mm])]

    doc.build(story)


def has_metrics(call):
    return any(i.get("type") == "metrics" for i in (call.get("flow_trace") or []))


def pdf_for_call(call, call_id, path):
    """Write the latency PDF for one already-fetched call record. Returns the analysis (None if no metrics)."""
    res = analyse(call)
    turns = [(call_id, i, t) for i, t in enumerate(res["turns"], 1)] if res else []
    write_pdf(str(path), [(call_id, res, None)], turns)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("call_ids", nargs="*")
    ap.add_argument("--file", action="append", default=[], help="saved /v1/calls/<id> JSON response")
    ap.add_argument("--save-dir", help="also save each fetched call JSON here")
    ap.add_argument("--pdf", help="PDF report path (default: latency_report_<timestamp>.pdf)")
    ap.add_argument("--no-pdf", action="store_true", help="console output only")
    opts = ap.parse_args()

    calls, report = [], []
    for f in opts.file:
        with open(f, encoding="utf-8") as fh:
            calls.append((os.path.basename(f), json.load(fh)))
    if opts.call_ids:
        token, org_id = login(), os.environ.get("LK_ORG_ID")
        for cid in opts.call_ids:
            try:
                data = fetch_call(token, cid, org_id)
            except Exception as e:  # report and continue with the other calls
                print(f"\n=== Call {cid}: FAILED to fetch: {e}")
                report.append((cid, None, str(e)))
                continue
            if opts.save_dir:
                os.makedirs(opts.save_dir, exist_ok=True)
                with open(os.path.join(opts.save_dir, f"{cid}.json"), "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2)
            calls.append((cid, data))
    if not calls and not report:
        sys.exit("No calls given.")

    all_turns = []
    for cid, call in calls:
        print(f"\n=== Call {cid}")
        res = analyse(call)
        report.append((cid, res, None))
        if res is None:
            print("No 'metrics' items in flow_trace - skipped.")
            continue
        turns = res["turns"]
        if res["user_lines"] != res["eou_count"]:
            print(f"NOTE: {res['eou_count']} EOU items vs {res['user_lines']} user transcript lines - caller text may be misaligned.")
        print(f"{'#':>3}  {'node_id':<22} {'caller text':<45} {'eou_ms':>7} {'#llm':>4} {'model_ms':>8} {'tts_ms':>7} {'TOTAL_ms':>9}")
        for i, t in enumerate(turns, 1):
            flag = "  <-- SLOW" if t["total"] > SLOW_MS else ""
            print(f"{i:>3}  {short(t['node_id'], 22):<22} {short(t['text']):<45} {t['eou']:>7.0f} {len(t['llm']):>4} "
                  f"{sum(t['llm']):>8.0f} {t['tts']:>7.0f} {t['total']:>9.0f}{flag}")
        if res["unanswered"]:
            print("Not counted (agent did not reply to this EOU):")
            for u in res["unanswered"]:
                print(f"     node={u['node_id']} eou={u['eou']:.0f}ms at={u['at']} text=\"{short(u['text'], 60)}\" ({u['reason']})")
        if turns:
            totals = [t["total"] for t in turns]
            slowest = max(turns, key=lambda t: t["total"])
            print(f"median {statistics.median(totals):.0f} ms | average {statistics.mean(totals):.0f} ms | "
                  f"slowest {slowest['total']:.0f} ms (turn {turns.index(slowest) + 1}, {slowest['node_id']})")
            print_latency_block(turns)
        else:
            print("No answered caller turns.")
        all_turns += [(cid, i, t) for i, t in enumerate(turns, 1)]

    if all_turns:
        print_overall(all_turns)
    if not opts.no_pdf:
        path = opts.pdf or f"latency_report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        try:
            write_pdf(path, report, all_turns)
        except RuntimeError as e:
            sys.exit(f"{e}   (or run with --no-pdf)")
        print(f"\nPDF report: {os.path.abspath(path)}")


def print_overall(all_turns):
    totals = [t["total"] for _, _, t in all_turns]
    slow = [x for x in all_turns if x[2]["total"] > SLOW_MS]
    print(f"\n=== All calls: {len(totals)} turns")
    print(f"median {statistics.median(totals):.0f} ms | average {statistics.mean(totals):.0f} ms | "
          f"p90 {pct(totals, 90):.0f} ms | above 3 s: {len(slow)} of {len(totals)}")
    print_latency_block([t for _, _, t in all_turns])
    if slow:
        print("\nSlow turns (> 3 s) - biggest part:")
        for cid, i, t in slow:
            print(f"  {cid} turn {i} ({t['node_id']}): {t['total']:.0f} ms -> {biggest_part(t)} "
                  f"[eou {t['eou']:.0f}, model {sum(t['llm']):.0f} in {len(t['llm'])} call(s), tts {t['tts']:.0f}]")


if __name__ == "__main__":
    main()
