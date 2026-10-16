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
import glob
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
TARGET_MS = int(os.environ.get("LK_TARGET_MS", "2500"))  # team target for real wait; a call's P90 above this fails


def http(method, path, token=None, data=None, org_id=None):
    headers = {}
    host = os.environ.get("WRAPPER_HOST", "").strip()  # tenant is chosen by Host; needed when BASE_URL is an IP
    if host:
        headers["Host"] = host
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


def clock_ms(eou_at, tts_at, eou_ms):
    """Real wait by the server clock: caller stopped (EOU time - EOU delay) -> first agent audio (TTS time)."""
    try:
        return (datetime.fromisoformat(tts_at) - datetime.fromisoformat(eou_at)).total_seconds() * 1000 + eou_ms
    except (TypeError, ValueError):
        return None


def opt_num(args, key):
    try:
        return num(args, key)
    except (KeyError, TypeError, ValueError):  # missing, or redacted to "<redacted>" in old results
        return 0


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
    print(f"  Real wait (clock)     {summary([t['wait'] for t in turns])}")
    print(f"  Sum of parts          {summary([t['total'] for t in turns])}")
    print(f"  Transcription latency {summary(stt) if stt else 'n/a (no transcription delay field in EOUMetrics)'}")
    print(f"  LLM latency           {summary([sum(t['llm']) for t in turns])}")
    print(f"  LLM TTFT per call     {summary([ms for t in turns for ms in t['llm']])}")
    print(f"  LLM TTFT first call   {summary([t['llm'][0] for t in turns if t['llm']])}")
    print(f"  TTS latency           {summary([t['tts'] for t in turns])}")


def msg_text(t):
    return t.get("content") or t.get("text") or t.get("message") or ""


def user_texts(call):
    return [msg_text(t) for t in call.get("transcript") or [] if t.get("role") == "user"]


def agent_replies(call):
    """For each caller line, everything the agent said after it, up to the caller's next line."""
    out = []
    for t in call.get("transcript") or []:
        if t.get("role") == "user":
            out.append([])
        elif t.get("role") == "assistant" and out:
            out[-1].append(msg_text(t))
    return [" ".join(x) for x in out]


def analyse(call):
    metrics = [i for i in (call.get("flow_trace") or []) if i.get("type") == "metrics"]
    if not metrics:
        return None
    turns, unanswered, current = [], [], None
    caller_idx = 0  # every EOU is one caller utterance, so it consumes one user transcript line
    texts = user_texts(call)
    replies = agent_replies(call)

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
                "reply": replies[caller_idx] if caller_idx < len(replies) else "",
                "eou": num(a, "eou_delay_ms"),
                "stt": transcription_ms(a),
                "llm": [],
                "tok_in": 0,   # prompt tokens, summed over the turn's model calls
                "tok_out": 0,  # generated tokens (total - prompt)
            }
            caller_idx += 1
        elif src == "LLMMetrics" and current is not None:
            current["llm"].append(num(a, "llm_ttft_ms"))
            prompt, total = opt_num(a, "llm_prompt_tokens"), opt_num(a, "llm_total_tokens")
            current["tok_in"] += prompt
            current["tok_out"] += max(total - prompt, 0)
        elif src == "TTSMetrics" and current is not None:
            tts = num(a, "tts_ttfb_ms")
            current["tts"] = tts
            current["total"] = current["eou"] + sum(current["llm"]) + tts  # sum of parts
            current["clock"] = clock_ms(current["at"], a.get("at"), current["eou"])
            current["wait"] = current["clock"] if current["clock"] is not None else current["total"]
            turns.append(current)
            current = None
        # STTMetrics, and LLM/TTS with no open caller turn (greeting / follow-on speech), are ignored
    if current is not None:
        close_unanswered("call ended before any TTS")
    for text in texts[caller_idx:]:  # caller lines the engine never recorded an end-of-turn for
        unanswered.append({"node_id": "-", "at": None, "text": text, "eou": None,
                           "reason": "no EOU recorded - agent never processed this line"})
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


def stage_split(t):
    """Where one turn's real wait went: EOU (includes STT), LLM (all model calls), TTS, and the rest (tools, gaps)."""
    llm = sum(t["llm"])
    return {"EOU": t["eou"], "LLM": llm, "TTS": t["tts"], "Other": max(t["wait"] - t["eou"] - llm - t["tts"], 0)}


def p90_report(all_turns, stt_by_call=None):
    """P90 view of a run: the run's P90, P90 per call, the turns slower than the run P90, and summary lines.
    stt_by_call: {call id: [STT ms per turn]} measured by the test bot (webcall_load_test.py), if available."""
    stt_by_call = {k: v for k, v in (stt_by_call or {}).items() if v}
    waits = [t["wait"] for _, _, t in all_turns]
    run_p90 = pct(waits, 90)
    calls = {}
    for cid, i, t in all_turns:
        calls.setdefault(cid, []).append((i, t))
    per_call = []
    for cid, items in calls.items():
        ts = [t for _, t in items]
        per_call.append({
            "cid": cid, "turns": len(ts), "p90": pct([t["wait"] for t in ts], 90),
            "median": statistics.median([t["wait"] for t in ts]),
            "above": sum(1 for t in ts if t["wait"] > run_p90),
            "eou_p90": pct([t["eou"] for t in ts], 90),
            "ttft_p90": pct([ms for t in ts for ms in t["llm"]], 90) if any(t["llm"] for t in ts) else 0,
            "tts_p90": pct([t["tts"] for t in ts], 90),
            "sum_p90": pct([t["total"] for t in ts], 90),
            "stt_p90": pct(stt_by_call[cid], 90) if cid in stt_by_call else None,
        })
    all_stt = [ms for v in stt_by_call.values() for ms in v]
    run = {"median": statistics.median(waits), "sum_p90": pct([t["total"] for _, _, t in all_turns], 90),
           "sum_median": statistics.median([t["total"] for _, _, t in all_turns]),
           "eou_p90": pct([t["eou"] for _, _, t in all_turns], 90),
           "ttft_p90": pct([ms for _, _, t in all_turns for ms in t["llm"]] or [0], 90),
           "tts_p90": pct([t["tts"] for _, _, t in all_turns], 90),
           "stt_p90": pct(all_stt, 90) if all_stt else None,
           "stt_median": statistics.median(all_stt) if all_stt else None}
    tail = sorted([x for x in all_turns if x[2]["wait"] > run_p90], key=lambda x: -x[2]["wait"])
    avg = {k: statistics.mean(stage_split(t)[k] for _, _, t in all_turns) for k in ("EOU", "LLM", "TTS", "Other")}
    mean_wait = statistics.mean(waits)
    share = {k: (v / mean_wait if mean_wait else 0) for k, v in avg.items()}
    main = max(share, key=share.get)
    biggest = [max(stage_split(t).items(), key=lambda kv: kv[1])[0] for _, _, t in all_turns]
    failing = [c for c in per_call if c["p90"] > TARGET_MS]
    tail_calls = sorted({cid for cid, _, _ in tail})
    lines = [
        f"Run P90 real wait is {run_p90:,.0f} ms against a target of {TARGET_MS:,} ms - "
        + ("PASS." if run_p90 <= TARGET_MS else f"FAIL ({run_p90 / TARGET_MS:.1f}x the target).")
        + f" Median real wait {run['median']:,.0f} ms.",
        f"Sum of parts (EOU + LLM + TTS, what the agent's own timers add up to): P90 {run['sum_p90']:,.0f} ms, "
        f"median {run['sum_median']:,.0f} ms - {run_p90 - run['sum_p90']:,.0f} ms below real wait at P90 "
        f"(the rest is tool calls and internal gaps).",
        f"{len(failing)} of {len(per_call)} call(s) have their own P90 above {TARGET_MS:,} ms; "
        f"{sum(1 for w in waits if w > TARGET_MS)} of {len(waits)} turns are over the target.",
        f"The slowest 10% ({len(tail)} turn(s) above {run_p90:,.0f} ms) come from {len(tail_calls)} call(s): "
        + ", ".join(f"{c[:8]} ({sum(1 for x in tail if x[0] == c)})" for c in tail_calls) + ".",
        f"Main bottleneck: {main} - {share[main]:.0%} of the average wait "
        f"(EOU {share['EOU']:.0%}, LLM {share['LLM']:.0%}, TTS {share['TTS']:.0%}, other {share['Other']:.0%}); "
        f"{main} is the biggest part in {biggest.count(main)} of {len(biggest)} turns.",
    ]
    if tail:
        cid, i, t = tail[0]
        lines.append(f"Slowest turn: {t['wait']:,.0f} ms (call {cid[:8]} turn {i}, {t['node_id']}) - "
                     f"{len(t['llm'])} model call(s) took {sum(t['llm']):,.0f} ms.")
    if all_stt:
        lines.append(f"STT (measured by the test bot: caller's voice stops -> final transcript): median "
                     f"{run['stt_median']:,.0f} ms, P90 {run['stt_p90']:,.0f} ms over {len(all_stt)} turns. "
                     f"STT runs inside the EOU wait, so it is part of EOU, not added on top.")
    else:
        lines.append("STT time is not reported by the platform (only audio length), so it is counted inside EOU.")
    return {"p90": run_p90, "per_call": per_call, "run": run, "tail": tail, "avg": avg, "share": share,
            "lines": lines}


def short(s, n=45):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def stats_row(name, values, unit=" ms"):
    if not values:
        return [name, "n/a", "n/a", "n/a", "n/a"]
    return [name, f"{pct(values, 90):,.0f}{unit}", f"{statistics.median(values):,.0f}{unit}",
            f"{statistics.mean(values):,.0f}{unit}", f"{min(values):,.0f}{unit}"]


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
                     stats_row("Real wait (server clock)", [t["wait"] for t in turns]),
                     stats_row("Sum of parts (EOU+LLM+TTS)", [t["total"] for t in turns]),
                     stats_row("Transcription", stt),
                     stats_row("LLM (all model calls)", [sum(t["llm"]) for t in turns]),
                     stats_row("LLM TTFT per model call", [ms for t in turns for ms in t["llm"]]),
                     stats_row("LLM TTFT first model call", [t["llm"][0] for t in turns if t["llm"]]),
                     stats_row("TTS (first audio)", [t["tts"] for t in turns]),
                     stats_row("LLM prompt tokens per turn", [t["tok_in"] for t in turns], ""),
                     stats_row("LLM output tokens per turn", [t["tok_out"] for t in turns], ""),
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
                              f"real wait = server clock, caller stops -> first agent audio | sum of parts = EOU + model TTFT + TTS TTFB | slow = real wait > {SLOW_MS} ms"),
                       small), Spacer(1, 4 * mm)]

    if all_turns:
        totals = [t["wait"] for _, _, t in all_turns]
        n_slow = sum(1 for v in totals if v > SLOW_MS)
        story += [Paragraph(f"All calls: {len(report)} call(s), {len(totals)} answered caller turns, "
                            f"<font color='#c0392b'><b>{n_slow} above 3 s</b></font>", ss["Heading2"]),
                  latency_table([t for _, _, t in all_turns]), Spacer(1, 4 * mm)]
        slow = [x for x in all_turns if x[2]["wait"] > SLOW_MS]
        if slow:
            rows = [["Call", "Turn", "Node", "Real wait", "Sum", "EOU wait", "LLM", "# model calls", "TTS", "Biggest part"]]
            for cid, i, t in slow:
                rows.append([short(cid, 30), i, short(t["node_id"], 24), f"{t['wait']:.0f}", f"{t['total']:.0f}",
                             f"{t['eou']:.0f}", f"{sum(t['llm']):.0f}", len(t["llm"]), f"{t['tts']:.0f}", biggest_part(t)])
            story += [Paragraph("Slow turns (real wait > 3 s)", ss["Heading3"]),
                      grid(rows, [54 * mm, 11 * mm, 42 * mm, 17 * mm, 15 * mm, 17 * mm, 15 * mm, 21 * mm, 13 * mm, 65 * mm])]

    def p90_section(p):
        """Separate page: P90 per call, the turns in the slowest 10%, and a summary of the run."""
        out = [PageBreak(), Paragraph("P90 report - slowest 10% of replies", ss["Heading2"]),
               Paragraph(f"P90 = 90% of caller turns got a reply this fast or faster. Target: real wait P90 "
                         f"&lt;= {TARGET_MS:,} ms per call (red = above). LLM TTFT is per model call.", small),
               Spacer(1, 3 * mm)]
        ms = lambda v: "n/a" if v is None else f"{v:,.0f}"  # noqa: E731
        rows = [["Call", "Turns", "Real wait P90", "Real wait median", "Sum of parts P90", "Turns > run P90",
                 "EOU P90", "STT P90", "LLM TTFT P90", "TTS P90", "Result"]]
        red_rows = []
        for n, c in enumerate(p["per_call"], 1):
            fail = c["p90"] > TARGET_MS
            rows.append([short(c["cid"], 26), c["turns"], ms(c["p90"]), ms(c["median"]), ms(c["sum_p90"]), c["above"],
                         ms(c["eou_p90"]), ms(c["stt_p90"]), ms(c["ttft_p90"]), ms(c["tts_p90"]),
                         "FAIL" if fail else "PASS"])
            if fail:
                red_rows.append(n)
        run = p["run"]
        rows.append(["All calls (run)", sum(c["turns"] for c in p["per_call"]), ms(p["p90"]), ms(run["median"]),
                     ms(run["sum_p90"]), len(p["tail"]), ms(run["eou_p90"]), ms(run["stt_p90"]), ms(run["ttft_p90"]),
                     ms(run["tts_p90"]), "FAIL" if p["p90"] > TARGET_MS else "PASS"])
        if p["p90"] > TARGET_MS:
            red_rows.append(len(rows) - 1)
        out += [Paragraph("P90 per call (all values in ms)", ss["Heading3"]),
                grid(rows, [48 * mm, 13 * mm, 22 * mm, 25 * mm, 25 * mm, 27 * mm, 17 * mm, 17 * mm, 23 * mm, 17 * mm, 17 * mm],
                     red_rows),
                Spacer(1, 4 * mm)]
        if p["tail"]:
            rows = [["Call", "Turn", "Node", "Caller said", "Real wait", "EOU", "LLM", "# model", "TTS", "Other", "Biggest"]]
            for cid, i, t in p["tail"]:
                s = stage_split(t)
                rows.append([short(cid, 22), i, Paragraph(escape(str(t["node_id"])), small),
                             Paragraph(escape(t["text"]), small), f"{t['wait']:,.0f}", f"{s['EOU']:,.0f}",
                             f"{s['LLM']:,.0f}", len(t["llm"]), f"{s['TTS']:,.0f}", f"{s['Other']:,.0f}",
                             max(s, key=s.get)])
            out += [Paragraph(f"Turns in the slowest 10% (real wait above run P90 {p['p90']:,.0f} ms)", ss["Heading3"]),
                    grid(rows, [40 * mm, 11 * mm, 36 * mm, 66 * mm, 18 * mm, 15 * mm, 16 * mm, 15 * mm, 13 * mm, 15 * mm, 18 * mm]),
                    Spacer(1, 4 * mm)]
        out.append(Paragraph("Summary", ss["Heading3"]))
        out += [Paragraph(f"{n}. {escape(line)}", ss["BodyText"]) for n, line in enumerate(p["lines"], 1)]
        return out

    if all_turns:
        stt_by_call = {cid: [t["stt_bot_ms"] for t in (res.get("caller_turns") or []) if t.get("stt_bot_ms") is not None]
                       for cid, res, _ in report if res}
        story += p90_section(p90_report(all_turns, stt_by_call))

    def caller_gap_section(caller_turns):
        """Agent stops talking -> caller starts talking, as measured by the test bot (webcall_load_test.py)."""
        bot = [t for t in caller_turns or [] if isinstance(t.get("caller_timing"), dict)]
        if not bot:
            return []
        def gap(t, k):
            c = t["caller_timing"]
            if k == "llm_ms":  # older results counted the hold inside llm_ms; derive it so the parts add up
                return max((c.get("caller_gap_ms") or 0) - (c.get("detect_wait_ms") or 0) - (c.get("tts_ms") or 0), 0)
            return c.get(k) or 0
        summary = grid([["Caller response time", "P90", "Median", "Average", "Min"],
                        stats_row("Total gap (agent stops -> caller speaks)", [gap(t, "caller_gap_ms") for t in bot]),
                        stats_row("  detect agent finished", [gap(t, "detect_wait_ms") for t in bot]),
                        stats_row("  caller bot thinking (LLM)", [gap(t, "llm_ms") for t in bot]),
                        stats_row("  caller speech generation", [gap(t, "tts_ms") for t in bot])],
                       [70 * mm, 28 * mm, 28 * mm, 28 * mm, 28 * mm])
        rows = [["#", "Agent said (caller answered this)", "Caller said", "Detect", "LLM", "Speech", "Gap ms"]]
        for t in bot:
            rows.append([t.get("turn"), Paragraph(escape(t.get("agent_text") or "(greeting / from audio)"), small),
                         Paragraph(escape(t.get("caller_spoken") or ""), small),
                         gap(t, "detect_wait_ms"), gap(t, "llm_ms"), gap(t, "tts_ms"), gap(t, "caller_gap_ms")])
        return [Spacer(1, 4 * mm),
                Paragraph("Caller response time - agent stops talking -> caller starts talking", ss["Heading3"]),
                Paragraph("Measured by the test bot, not the server. This is the simulated caller's own delay "
                          "(waiting to be sure the agent finished + its LLM + speech synthesis); "
                          "a real person's gap will differ.", small),
                Spacer(1, 2 * mm), summary, Spacer(1, 3 * mm),
                grid(rows, [10 * mm, 112 * mm, 70 * mm, 18 * mm, 18 * mm, 18 * mm, 20 * mm])]

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
            rows = [["#", "Node", "Caller said", "Agent replied", "EOU ms", "# model", "LLM ms", "Tokens in / out",
                     "TTS ms", "Sum ms", "Real wait ms"]]
            slow_rows = []
            for i, t in enumerate(turns, 1):
                rows.append([i, Paragraph(escape(str(t["node_id"])), small), Paragraph(escape(t["text"]), small),
                             Paragraph(escape(t.get("reply") or "-"), small),
                             f"{t['eou']:.0f}", len(t["llm"]), f"{sum(t['llm']):.0f}",
                             f"{t['tok_in']:,.0f} / {t['tok_out']:,.0f}", f"{t['tts']:.0f}",
                             f"{t['total']:.0f}",
                             f"{t['wait']:.0f}" + (" SLOW" if t["wait"] > SLOW_MS else "")])
                if t["wait"] > SLOW_MS:
                    slow_rows.append(i)
            story.append(grid(rows, [8 * mm, 28 * mm, 46 * mm, 66 * mm, 13 * mm, 13 * mm, 14 * mm, 26 * mm, 13 * mm, 15 * mm, 22 * mm],
                              slow_rows))
        if res["unanswered"]:
            rows = [["Node", "Caller text", "EOU ms", "Reason"]]
            for u in res["unanswered"]:
                rows.append([Paragraph(escape(str(u["node_id"])), small), Paragraph(escape(u["text"]), small),
                             "-" if u['eou'] is None else f"{u['eou']:.0f}", u["reason"]])
            story += [Paragraph("Not counted - agent did not reply to this EOU", ss["Heading3"]),
                      grid(rows, [45 * mm, 140 * mm, 20 * mm, 55 * mm])]
        story += caller_gap_section(res.get("caller_turns"))

    doc.build(story)


def has_metrics(call):
    return any(i.get("type") == "metrics" for i in (call.get("flow_trace") or []))


def pdf_for_call(call, call_id, path, caller_turns=None):
    """Write the latency PDF for one already-fetched call record. Returns the analysis (None if no metrics).
    caller_turns: the test bot's per-turn results, which add the caller response time section."""
    res = analyse(call)
    if res is not None:
        res["caller_turns"] = caller_turns
    turns = [(call_id, i, t) for i, t in enumerate(res["turns"], 1)] if res else []
    write_pdf(str(path), [(call_id, res, None)], turns)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("call_ids", nargs="*")
    ap.add_argument("--file", action="extend", nargs="+", default=[],
                    help="saved /v1/calls/<id> JSON, or a webcall_load_test.py result JSON (adds caller response time)")
    ap.add_argument("--save-dir", help="also save each fetched call JSON here")
    ap.add_argument("--pdf", help="PDF report path (default: latency_report_<timestamp>.pdf)")
    ap.add_argument("--no-pdf", action="store_true", help="console output only")
    opts = ap.parse_args()

    calls, report, bot_turns = [], [], {}
    # Windows shells do not expand wildcards, so expand results\*.json here
    files = [m for f in opts.file for m in (sorted(glob.glob(f)) or [f])]
    for f in files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        if "call_record" in data:  # test-bot result file: the call record plus the bot's own timings
            cid = data.get("call_id") or os.path.basename(f)
            bot_turns[cid] = data.get("turns")
            calls.append((cid, data["call_record"]))
        else:
            calls.append((os.path.basename(f), data))
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
        if res is not None:
            res["caller_turns"] = bot_turns.get(cid)
        report.append((cid, res, None))
        if res is None:
            print("No 'metrics' items in flow_trace - skipped.")
            continue
        turns = res["turns"]
        if res["user_lines"] != res["eou_count"]:
            print(f"NOTE: {res['eou_count']} EOU items vs {res['user_lines']} user transcript lines - caller text may be misaligned.")
        print(f"{'#':>3}  {'node_id':<22} {'caller text':<45} {'eou_ms':>7} {'#llm':>4} {'model_ms':>8} {'tts_ms':>7} {'sum_ms':>7} {'real_ms':>8}")
        for i, t in enumerate(turns, 1):
            flag = "  <-- SLOW" if t["wait"] > SLOW_MS else ""
            print(f"{i:>3}  {short(t['node_id'], 22):<22} {short(t['text']):<45} {t['eou']:>7.0f} {len(t['llm']):>4} "
                  f"{sum(t['llm']):>8.0f} {t['tts']:>7.0f} {t['total']:>7.0f} {t['wait']:>8.0f}{flag}")
        if res["unanswered"]:
            print("Not counted (agent did not reply to this EOU):")
            for u in res["unanswered"]:
                eou = "-" if u["eou"] is None else f"{u['eou']:.0f}ms"
                print(f"     node={u['node_id']} eou={eou} at={u['at']} text=\"{short(u['text'], 60)}\" ({u['reason']})")
        if turns:
            totals = [t["wait"] for t in turns]
            slowest = max(turns, key=lambda t: t["wait"])
            print(f"real wait: median {statistics.median(totals):.0f} ms | average {statistics.mean(totals):.0f} ms | "
                  f"slowest {slowest['wait']:.0f} ms (turn {turns.index(slowest) + 1}, {slowest['node_id']})")
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
    totals = [t["wait"] for _, _, t in all_turns]
    slow = [x for x in all_turns if x[2]["wait"] > SLOW_MS]
    print(f"\n=== All calls: {len(totals)} turns")
    print(f"real wait: median {statistics.median(totals):.0f} ms | average {statistics.mean(totals):.0f} ms | "
          f"p90 {pct(totals, 90):.0f} ms | above 3 s: {len(slow)} of {len(totals)}")
    print_latency_block([t for _, _, t in all_turns])
    p = p90_report(all_turns)
    print(f"\nP90 per call (target {TARGET_MS} ms):")
    for c in p["per_call"]:
        print(f"  {short(c['cid'], 34):<34} turns {c['turns']:>3}  P90 {c['p90']:>6.0f} ms  above run P90 {c['above']:>2}  "
              f"EOU P90 {c['eou_p90']:>5.0f}  TTFT P90 {c['ttft_p90']:>5.0f}  TTS P90 {c['tts_p90']:>4.0f}  "
              f"{'FAIL' if c['p90'] > TARGET_MS else 'PASS'}")
    print("P90 summary:")
    for n, line in enumerate(p["lines"], 1):
        print(f"  {n}. {line}")
    if slow:
        print("\nSlow turns (> 3 s) - biggest part:")
        for cid, i, t in slow:
            print(f"  {cid} turn {i} ({t['node_id']}): real {t['wait']:.0f} ms, sum {t['total']:.0f} ms -> {biggest_part(t)} "
                  f"[eou {t['eou']:.0f}, model {sum(t['llm']):.0f} in {len(t['llm'])} call(s), tts {t['tts']:.0f}]")


if __name__ == "__main__":
    main()
