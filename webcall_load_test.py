#!/usr/bin/env python3
"""
AI Labs web-call test (one simulated user)

Flow per user:
    login -> POST /v1/agents/{id}/web-calls -> join LiveKit room as the caller
    -> wait for the agent's greeting -> speak a recorded question
    -> measure time until the agent replies -> hang up
    -> fetch GET /v1/calls/{call_id} for server-side per-turn timings

Config is read from .env:
    WRAPPER_BASE_URL   e.g. http://172.17.20.214:6002
    WRAPPER_USERNAME   login email
    WRAPPER_PASSWORD   login password
    AGENT_ID           agent to call
    WRAPPER_HOST       optional: Host header for every AI Labs API request (the server picks the tenant
                       from it), e.g. ailabs.healthviewx.com when calling the server by IP

Usage:
    python webcall_load_test.py --question uploads/question.wav
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import statistics
import subprocess
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
from livekit import rtc

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# The question the test caller asks when running in single-question mode.
DEFAULT_QUESTION_TEXT = "Hello. Can you tell me what you can help me with today?"

PRESET_SCENARIOS = {
    "booking": [
        "Hi, I'd like to book a new appointment.",
        "{first_name}.",
        "{last_name}.",
        "{dob}.",
        "Yes, that is correct.",
        "I need a {specialty} appointment next Monday morning.",
        "Yes, that's right.",
        "The first option, please.",
        "Yes, please book that.",
        "No, that's all. Thank you."
    ]
}

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
SPEECH_RMS = 400          # int16 RMS above this counts as agent speech
END_OF_SPEECH_S = 1.2     # silence this long ends an agent utterance


def load_env(path=".env"):
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def host_header():
    """Host header for AI Labs API requests when WRAPPER_HOST is set (tenant is chosen by host)."""
    host = os.environ.get("WRAPPER_HOST", "").strip()
    return {"Host": host} if host else {}


def find_key(obj, names):
    """Return the first value whose key is in names, searching nested dicts."""
    if isinstance(obj, dict):
        for n in names:
            if obj.get(n):
                return obj[n]
        for v in obj.values():
            found = find_key(v, names)
            if found:
                return found
    return None


def redact(obj):
    if isinstance(obj, dict):
        return {k: ("<redacted>" if isinstance(v, str) and ("token" in k.lower() or "secret" in k.lower())
                    else redact(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


async def fetch_call_record(session, base, headers, call_id, wait_for_metrics=20):
    """GET the call record; if its timing metrics are not stored yet, retry for up to wait_for_metrics seconds."""
    from call_latency import has_metrics
    deadline = time.monotonic() + wait_for_metrics
    while True:
        async with session.get(f"{base}/v1/calls/{call_id}", headers=headers) as resp:
            status, body = resp.status, await resp.json(content_type=None)
        if status >= 300 or has_metrics(body) or time.monotonic() > deadline:
            return status, body
        await asyncio.sleep(2)


def write_latency_pdf(call_id, record, out, caller_turns=None):
    """Latency PDF for this call next to the result JSON: results/<name>_latency_<call id>.pdf."""
    from call_latency import pdf_for_call
    path = Path(out).with_name(f"{Path(out).stem}_latency_{call_id}.pdf")
    try:
        res = pdf_for_call(record, call_id, path, caller_turns)
    except Exception as e:  # never fail the call test because of the report
        print(f"  latency PDF failed: {e}")
        return None
    if res is None:
        print("  latency PDF: no timing metrics in the call record yet (PDF says so)")
    else:
        totals = sorted(t["wait"] for t in res["turns"])  # real wait (server clock)
        if totals:
            print(f"  real wait: {len(totals)} turns, median {statistics.median(totals):.0f} ms, "
                  f"slowest {totals[-1]:.0f} ms, {sum(v > 3000 for v in totals)} above 3 s")
    print(f"Latency PDF: {path}")
    return str(path)


def watch_transcriptions(room, log, tasks):
    """Log the live transcriptions the agent publishes (LiveKit topic lk.transcription, or the older
    transcription_received event) with the bot's own clock, so STT time can be measured per turn."""
    def on_stream(reader, sender):
        async def read():
            text = await reader.read_all()
            a = reader.info.attributes or {}  # final flag can arrive in the stream trailer, so read it at the end
            log("transcription", sender=sender, track=a.get("lk.transcribed_track_id"),
                final=str(a.get("lk.transcription_final", "")).lower() == "true",
                segment=a.get("lk.segment_id"), text=text[:120])
        tasks.append(asyncio.create_task(read()))
    try:
        room.register_text_stream_handler("lk.transcription", on_stream)
    except Exception as e:
        log("transcription_watch_failed", error=str(e))
    room.on("transcription_received", lambda segments, participant, publication: [
        log("transcription", sender=getattr(participant, "identity", None), track=getattr(publication, "sid", None),
            final=bool(s.final), segment=s.id, text=s.text[:120]) for s in segments])


def trailing_silence_ms(samples, win_ms=20):
    """Silence after the last spoken word in the caller's clip (speech TTS pads the end), so STT can be timed
    from when the voice actually stops, not from when the whole clip has been sent."""
    win = SAMPLE_RATE * win_ms // 1000
    n = len(samples) // win
    if n == 0:
        return 0
    rms = np.sqrt((samples[:n * win].astype(np.float64).reshape(n, win) ** 2).mean(axis=1))
    voiced = np.nonzero(rms > max(300.0, 0.05 * rms.max()))[0]
    if voiced.size == 0:
        return 0
    return round((len(samples) - (voiced[-1] + 1) * win) / SAMPLE_RATE * 1000)


def add_bot_stt(result):
    """STT time per turn from the bot's clock: caller's voice stops -> last final transcript of the caller's words.
    Adds turn["stt_bot_ms"] and result["stt_bot"]; needs the agent to publish transcriptions to the room."""
    events = result.get("events") or []
    mic_sid = next((e.get("sid") for e in events if e["event"] == "mic_published"), None)
    finals = [e for e in events if e["event"] == "transcription" and e.get("final")
              and (mic_sid is None or e.get("track") == mic_sid)]
    starts = {e["turn"]: e["t_ms"] for e in events if e["event"] == "turn_user_speak_start"}
    # caller's voice stops = clip fully sent minus the clip's trailing silence
    ends = {e["turn"]: e["t_ms"] - (e.get("trailing_silence_ms") or 0)
            for e in events if e["event"] == "turn_user_speak_end"}
    by_turn = {}
    for n, t_end in ends.items():
        nxt = min([t for k, t in starts.items() if k > n] or [float("inf")])
        hits = [e["t_ms"] for e in finals if starts.get(n, t_end) <= e["t_ms"] < nxt]
        if hits:
            by_turn[n] = max(hits) - t_end
    for t in result.get("turns") or []:
        if t.get("turn") in by_turn:
            t["stt_bot_ms"] = by_turn[t["turn"]]
    seen = sum(1 for e in events if e["event"] == "transcription")
    result["stt_bot"] = {"transcription_events": seen, "final_caller_transcripts": len(finals),
                         "turns_measured": len(by_turn), "stt_ms_by_turn": by_turn}
    if by_turn:
        v = sorted(by_turn.values())
        print(f"  STT (bot clock, caller stops -> final transcript): {len(v)} turns, "
              f"median {statistics.median(v):.0f} ms, max {v[-1]:.0f} ms")
    else:
        print(f"  STT (bot clock): not measurable - {seen} transcription event(s) received from the room"
              + (" (agent does not publish transcriptions)" if not seen else ""))


class SpeechWorker:
    """One long-running PowerShell speech process, so each turn skips the ~1 s PowerShell start-up."""

    SCRIPT = (
        "Add-Type -AssemblyName System.Speech;"
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,"
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono);"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "while (($l = [Console]::In.ReadLine()) -ne $null) {"
        " $p = $l.Split('|');"
        " $o = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($p[0]));"
        " $t = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($p[1]));"
        " $s.SetOutputToWaveFile($o, $f); $s.Speak($t); $s.SetOutputToNull();"
        " [Console]::Out.WriteLine('OK'); [Console]::Out.Flush() }"
    )

    def __init__(self):
        self.proc = subprocess.Popen(["powershell", "-NoProfile", "-Command", self.SCRIPT],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def speak(self, text, path):
        b64 = lambda s: base64.b64encode(s.encode("utf-8")).decode("ascii")
        self.proc.stdin.write(f"{b64(str(Path(path).resolve()))}|{b64(text)}\n")
        self.proc.stdin.flush()
        if self.proc.stdout.readline().strip() != "OK":
            raise RuntimeError("speech worker failed")

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)


_speech_worker = None
_speech_lock = threading.Lock()  # one speech process, shared by parallel calls: one request at a time


def text_to_wav(text, path):
    """Speak text with the Windows built-in voice into a 16 kHz mono 16-bit WAV."""
    with _speech_lock:
        _text_to_wav(text, path)


def _text_to_wav(text, path):
    global _speech_worker
    Path(path).parent.mkdir(exist_ok=True)
    try:
        if _speech_worker is None or _speech_worker.proc.poll() is not None:
            _speech_worker = SpeechWorker()
        _speech_worker.speak(text, path)
        return
    except Exception:
        _speech_worker = None  # fall back to a one-off PowerShell call below
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,"
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono);"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$s.SetOutputToWaveFile($env:Q_OUT, $f); $s.Speak($env:Q_TEXT); $s.Dispose()"
    )
    Path(path).parent.mkdir(exist_ok=True)
    env = {**os.environ, "Q_TEXT": text, "Q_OUT": str(Path(path).resolve())}
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, env=env)


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path} must be 16 kHz, mono, 16-bit")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


class Recorder:
    """Collects one continuous audio channel; written later aligned on a shared clock."""

    def __init__(self):
        self.start = None
        self.chunks = []

    def add(self, samples):
        if self.start is None:
            self.start = time.monotonic()
        self.chunks.append(samples.copy())

    def samples(self):
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.int16)


def _placed(rec, t_ref):
    s = rec.samples().astype(np.int32)
    offset = max(0, round((rec.start - t_ref) * SAMPLE_RATE)) if rec.start else 0
    return np.concatenate([np.zeros(offset, dtype=np.int32), s])


def write_stereo(path, t_ref, left, rights):
    """Left = caller (our question), right = agent (all its audio tracks mixed)."""
    lch = _placed(left, t_ref)
    rparts = [_placed(r, t_ref) for r in rights]
    n = max([len(lch)] + [len(x) for x in rparts])
    stereo = np.zeros((n, 2), dtype=np.int32)
    stereo[:len(lch), 0] = lch
    for x in rparts:
        stereo[:len(x), 1] += x
    stereo = np.clip(stereo, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(stereo.tobytes())


def extract_agent_audio(tracks, t_start, t_end, path):
    """Mix and save the agent audio recorded between t_start and t_end as a 16kHz mono WAV."""
    if not tracks:
        return False
    parts = []
    for tl in tracks:
        if not tl.recorder.chunks or tl.recorder.start is None:
            continue
        sample_start = max(0, int((t_start - tl.recorder.start) * SAMPLE_RATE))
        sample_end = max(sample_start, int((t_end - tl.recorder.start) * SAMPLE_RATE))
        all_samples = tl.recorder.samples()
        if sample_start < len(all_samples):
            parts.append(all_samples[sample_start:min(sample_end, len(all_samples))])
    if not parts:
        return False
    max_len = max(len(p) for p in parts)
    mixed = np.zeros(max_len, dtype=np.int32)
    for p in parts:
        mixed[:len(p)] += p
    mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(mixed.tobytes())
    return True


class TrackListener:
    """Speech detection + recording for one agent audio track."""

    def __init__(self):
        self.utterances = []      # list of [start, end]
        self._last_loud = None
        self.recorder = Recorder()

    async def consume(self, track):
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
        async for ev in stream:
            samples = np.frombuffer(ev.frame.data, dtype=np.int16)
            if samples.size == 0:
                continue
            self.recorder.add(samples)
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            now = time.monotonic()
            if rms >= SPEECH_RMS:
                if not self.utterances or self.utterances[-1][1] is not None:
                    self.utterances.append([now, None])
                self._last_loud = now
            elif self.utterances and self.utterances[-1][1] is None and now - self._last_loud >= END_OF_SPEECH_S:
                self.utterances[-1][1] = self._last_loud


class AgentListener:
    """Listens to every audio track the agent publishes (it may publish more than one)."""

    def __init__(self):
        self.tracks = []
        self.track_received_at = None
        self.closed = False  # set when the room disconnects: stop waiting for audio

    def consume(self, track):
        if self.track_received_at is None:
            self.track_received_at = time.monotonic()
        tl = TrackListener()
        self.tracks.append(tl)
        return tl.consume(track)

    @property
    def recorders(self):
        return [tl.recorder for tl in self.tracks]

    async def wait_for_utterance_start(self, after, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.closed:
            for tl in self.tracks:
                for u in tl.utterances:
                    if u[0] >= after:
                        return u
            await asyncio.sleep(0.02)
        return None

    async def wait_for_utterance_end(self, utt, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and utt[1] is None and not self.closed:
            await asyncio.sleep(0.05)
        return utt[1]

    async def wait_for_full_reply(self, after, reply_timeout, settle_time=1.5):
        """Wait for agent to reply, waiting out intermediate pauses (e.g. tool execution)."""
        first_utt = await self.wait_for_utterance_start(after, reply_timeout)
        if not first_utt:
            return None, None
        last_end = await self.wait_for_utterance_end(first_utt, 60)
        if not last_end:
            last_end = time.monotonic()

        while True:
            next_utt = await self.wait_for_utterance_start(last_end, settle_time)
            if not next_utt:
                break
            last_end = await self.wait_for_utterance_end(next_utt, 60)
            if not last_end:
                last_end = time.monotonic()

        return first_utt, last_end


class NoiseGenerator:
    """Generates synthetic background noise frames (16kHz mono int16) to test agent noise cancellation."""

    def __init__(self, noise_type="none", noise_level=0.0):
        self.noise_type = (noise_type or "none").lower()
        self.noise_level = max(0.0, min(1.0, float(noise_level)))
        self.sample_idx = 0
        self._next_click_sample = 0
        self._click_decay = 0.0

    def generate_frame(self, num_samples=FRAME_SAMPLES):
        if self.noise_type in ("none", "off", "") or self.noise_level <= 0.0:
            return np.zeros(num_samples, dtype=np.int16)

        t = (np.arange(num_samples) + self.sample_idx) / SAMPLE_RATE
        self.sample_idx += num_samples

        if self.noise_type in ("low", "fan", "hum", "ac"):
            # 1. Low Noise: Continuous stationary drone (60Hz + 120Hz + 180Hz hum + pink/room noise)
            hum = 0.6 * np.sin(2 * np.pi * 60 * t) + 0.3 * np.sin(2 * np.pi * 120 * t) + 0.1 * np.sin(2 * np.pi * 180 * t)
            white = np.random.normal(0, 0.2, num_samples)
            signal = hum + white
            amplitude = self.noise_level * 5000

        elif self.noise_type in ("medium", "typing", "clicks"):
            # 2. Medium Noise: Intermittent mechanical keyboard typing clicks with gentle baseline
            signal = np.random.normal(0, 0.05, num_samples)
            for i in range(num_samples):
                if self.sample_idx - num_samples + i >= self._next_click_sample:
                    self._click_decay = 1.0
                    self._next_click_sample = (self.sample_idx - num_samples + i) + int(np.random.uniform(0.08, 0.28) * SAMPLE_RATE)
                if self._click_decay > 0.01:
                    click_val = self._click_decay * np.sin(2 * np.pi * 1200 * (i / SAMPLE_RATE))
                    signal[i] += click_val * 2.5
                    self._click_decay *= 0.85
            amplitude = self.noise_level * 10000

        elif self.noise_type in ("high", "cafe", "babble", "chatter"):
            # 3. High Noise: Multi-speaker ambient cocktail babble / cafe chatter
            v1 = np.sin(2 * np.pi * (250 + 50 * np.sin(2 * np.pi * 1.5 * t)) * t) * np.sin(2 * np.pi * 3.0 * t)
            v2 = np.sin(2 * np.pi * (450 + 80 * np.sin(2 * np.pi * 2.2 * t)) * t) * np.sin(2 * np.pi * 4.5 * t)
            v3 = np.sin(2 * np.pi * (750 + 120 * np.sin(2 * np.pi * 0.8 * t)) * t) * np.sin(2 * np.pi * 2.0 * t)
            hiss = np.random.normal(0, 0.3, num_samples)
            signal = (v1 * 0.4 + v2 * 0.3 + v3 * 0.3) + hiss
            amplitude = self.noise_level * 8000
        else:
            signal = np.random.normal(0, 0.2, num_samples)
            amplitude = self.noise_level * 4000

        return np.clip(signal * amplitude, -32768, 32767).astype(np.int16)


class Mic:
    """Publishes a continuous mic track: silence (or noise), and speech (or mixed speech+noise) when asked."""

    def __init__(self, noise_type="none", noise_level=0.0):
        self.source = rtc.AudioSource(SAMPLE_RATE, 1)
        self._queue = None
        self._running = True
        self.speech_done = asyncio.Event()
        self.recorder = Recorder()
        self.noise_gen = NoiseGenerator(noise_type=noise_type, noise_level=noise_level)

    def say(self, samples):
        self.speech_done.clear()
        self._queue = samples

    async def run(self):
        next_t = time.monotonic()
        pos = 0
        while self._running:
            noise_frame = self.noise_gen.generate_frame(FRAME_SAMPLES)
            if self._queue is not None:
                chunk = self._queue[pos:pos + FRAME_SAMPLES]
                pos += FRAME_SAMPLES
                if chunk.size < FRAME_SAMPLES:
                    chunk = np.pad(chunk, (0, FRAME_SAMPLES - chunk.size))
                if pos >= self._queue.size:
                    self._queue, pos = None, 0
                    self.speech_done.set()
                # Mix caller speech with background noise
                mixed = chunk.astype(np.int32) + noise_frame.astype(np.int32)
                chunk = np.clip(mixed, -32768, 32767).astype(np.int16)
            else:
                chunk = noise_frame
            frame = rtc.AudioFrame.create(SAMPLE_RATE, 1, FRAME_SAMPLES)
            np.copyto(np.frombuffer(frame.data, dtype=np.int16), chunk)
            await self.source.capture_frame(frame)
            self.recorder.add(chunk)
            next_t += FRAME_MS / 1000
            await asyncio.sleep(max(0, next_t - time.monotonic()))

    def stop(self):
        self._running = False


async def run_one_user(session, base, auth_headers, agent_id, turns, greeting_timeout, reply_timeout,
                       livekit_url_override=None, audio_path=None, noise_type="none", noise_level=0.0):
    r = {
        "started_at": datetime.now().isoformat(),
        "success": False,
        "steps": {},
        "noise": {"type": noise_type, "level": noise_level} if noise_type != "none" else None,
    }
    t0 = time.monotonic()

    # 1. create the web call
    async with session.post(f"{base}/v1/agents/{agent_id}/web-calls", json={}, headers=auth_headers) as resp:
        body = await resp.json(content_type=None)
        r["steps"]["web_call_http_status"] = resp.status
        r["steps"]["web_call_ms"] = round((time.monotonic() - t0) * 1000)
        r["web_call_response"] = redact(body)
        if resp.status >= 300:
            r["error"] = f"web-calls returned HTTP {resp.status}"
            return r

    token = find_key(body, ["token", "join_token", "participant_token", "access_token"])
    url = find_key(body, ["livekit_url", "ws_url", "server_url", "url"])
    call_id = find_key(body, ["call_id", "id"])
    r["call_id"] = call_id
    r["livekit_url_returned"] = url
    if livekit_url_override:
        url = livekit_url_override
    r["livekit_url_used"] = url
    if not token or not url:
        r["error"] = "could not find join token / livekit_url in web-calls response"
        return r

    # 2. join the room
    room = rtc.Room()
    listener = AgentListener()
    mic = Mic(noise_type=noise_type, noise_level=noise_level)
    tasks = []

    events = r["events"] = []
    t_join = time.monotonic()

    def log(name, **kw):
        events.append({"t_ms": round((time.monotonic() - t_join) * 1000), "event": name, **kw})

    @room.on("track_subscribed")
    def _on_track(track, publication, participant):
        log("track_subscribed", participant=participant.identity, kind=str(track.kind))
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            r["agent_identity"] = participant.identity
            tasks.append(asyncio.create_task(listener.consume(track)))

    room.on("participant_connected", lambda p: log("participant_connected", participant=p.identity))
    room.on("participant_disconnected", lambda p: log("participant_disconnected", participant=p.identity))
    room.on("track_published", lambda pub, p: log("track_published", participant=p.identity, kind=str(pub.kind)))
    room.on("track_subscription_failed", lambda p, sid, err: log("track_subscription_failed", participant=p.identity, error=str(err)))
    room.on("connection_state_changed", lambda s: log("connection_state", state=str(s)))
    room.on("disconnected", lambda reason: (log("disconnected", reason=str(reason)),
                                           setattr(listener, "closed", True)))
    watch_transcriptions(room, log, tasks)

    try:
        t_join = time.monotonic()
        await room.connect(url, token)
        r["steps"]["room_join_ms"] = round((time.monotonic() - t_join) * 1000)
        r["room"] = room.name
        log("connected", already_in_room=[p.identity for p in room.remote_participants.values()])

        track = rtc.LocalAudioTrack.create_audio_track("mic", mic.source)
        mic_pub = await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        log("mic_published", sid=mic_pub.sid)
        tasks.append(asyncio.create_task(mic.run()))

        # 3. greeting
        greeting = await listener.wait_for_utterance_start(t_join, greeting_timeout)
        if listener.track_received_at:
            r["steps"]["agent_audio_track_ms"] = round((listener.track_received_at - t_join) * 1000)
        if not greeting:
            r["error"] = "silent call: agent never spoke (no agent joined, or worker overloaded)"
            return r
        r["steps"]["greeting_start_ms"] = round((greeting[0] - t_join) * 1000)
        r["steps"]["call_setup_ms"] = round((greeting[0] - t0) * 1000)  # start call -> agent voice heard
        end = await listener.wait_for_utterance_end(greeting, 30)
        if end:
            r["steps"]["greeting_duration_ms"] = round((end - greeting[0]) * 1000)
        await asyncio.sleep(0.6)

        # 4. conversational turns
        r["turns"] = []
        turn_latencies = []
        all_turns_ok = True

        for turn_idx, (turn_text, turn_samples) in enumerate(turns, start=1):
            turn_info = {
                "turn": turn_idx,
                "text": turn_text,
                "question_ms": round(len(turn_samples) / SAMPLE_RATE * 1000),
            }
            log("turn_user_speak_start", turn=turn_idx, text=turn_text)
            mic.say(turn_samples)
            await mic.speech_done.wait()
            t_spoken = time.monotonic()
            log("turn_user_speak_end", turn=turn_idx, trailing_silence_ms=trailing_silence_ms(turn_samples))

            first_utt, last_end = await listener.wait_for_full_reply(t_spoken, reply_timeout, settle_time=1.5)
            if not first_utt:
                # If caller wrapped up and agent ended call
                if turn_idx == len(turns) and room.connection_state == rtc.ConnectionState.CONN_DISCONNECTED:
                    log("room_disconnected", turn=turn_idx)
                    r["turns"].append(turn_info)
                    break
                turn_info["error"] = f"agent did not reply within {reply_timeout}s"
                r["turns"].append(turn_info)
                r["error"] = f"turn {turn_idx} ({turn_text!r}): agent did not reply within {reply_timeout}s"
                all_turns_ok = False
                break

            agent_resp_ms = round((first_utt[0] - t_spoken) * 1000)
            reply_dur_ms = round((last_end - first_utt[0]) * 1000)
            turn_info["agent_response_ms"] = agent_resp_ms
            turn_info["reply_duration_ms"] = reply_dur_ms
            turn_latencies.append(agent_resp_ms)
            log("turn_agent_reply_done", turn=turn_idx, latency_ms=agent_resp_ms, duration_ms=reply_dur_ms)
            r["turns"].append(turn_info)

            # Brief pause before next turn
            await asyncio.sleep(0.6)

        # Set top-level step metrics for backwards compatibility / reporting
        if r["turns"]:
            r["steps"]["question_ms"] = r["turns"][0]["question_ms"]
            r["steps"]["agent_response_ms"] = r["turns"][0].get("agent_response_ms", 0)
            if "reply_duration_ms" in r["turns"][0]:
                r["steps"]["reply_duration_ms"] = r["turns"][0]["reply_duration_ms"]
            if turn_latencies:
                r["steps"]["avg_agent_response_ms"] = round(sum(turn_latencies) / len(turn_latencies))

        if all_turns_ok:
            r["success"] = True
        return r
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
        msg = str(e)
        if "parse the url" in msg or "UrlParse" in msg:
            r["error"] += f" | The LiveKit URL is not valid: {url!r}. Check the --livekit-url value."
        elif "wait_pc_connection" in msg:
            r["error"] += (" | Could not open the audio connection to LiveKit. Signalling works but WebRTC media "
                           "(UDP 50000-50200, TCP 7881 fallback) is blocked from this network. "
                           "Use the network/VPN where tests worked, or ask DevOps to open those ports.")
        return r
    finally:
        mic.stop()
        for t in tasks:
            t.cancel()
        await room.disconnect()
        if audio_path and (mic.recorder.chunks or any(x.chunks for x in listener.recorders)):
            write_stereo(audio_path, t_join, mic.recorder, listener.recorders)
            r["audio_file"] = str(audio_path)
        r["steps"]["total_ms"] = round((time.monotonic() - t0) * 1000)


async def gemini_generate_reply(session, api_key, model, fallback_models, history, system_instruction,
                                info=None, request_timeout=10):
    """info (dict, optional) is filled with the model that answered and any failed attempts."""
    info = info if info is not None else {}
    info["attempts"] = []
    models = [model] + [m.strip() for m in fallback_models.split(",") if m.strip()]
    payload = {
        "system_instruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": history,
        "generationConfig": {
            "maxOutputTokens": 50,
            "temperature": 0.2
        }
    }
    for m in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
        t = time.monotonic()
        try:
            async with session.post(url, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=request_timeout)) as resp:
                ms = round((time.monotonic() - t) * 1000)
                if resp.status == 200:
                    data = await resp.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            info["model"] = m
                            return parts[0]["text"].strip()
                info["attempts"].append({"model": m, "status": resp.status, "ms": ms})
        except Exception as e:
            info["attempts"].append({"model": m, "status": type(e).__name__,
                                     "ms": round((time.monotonic() - t) * 1000)})
    raise RuntimeError("All Gemini models failed or were rate limited.")


async def fetch_new_agent_text(session, base, auth_headers, call_id, seen):
    """Return agent messages added to the call transcript since the last call (None if unavailable)."""
    try:
        async with session.get(f"{base}/v1/calls/{call_id}", headers=auth_headers,
                               timeout=aiohttp.ClientTimeout(total=3)) as resp:
            if resp.status != 200:
                return None
            rec = await resp.json(content_type=None)
    except Exception:
        return None
    said = [x.get("content") or "" for x in rec.get("transcript", []) if x.get("role") == "assistant"]
    new = said[seen[0]:]
    if not new:
        return None
    seen[0] = len(said)
    return " ".join(new).replace("—", "-").replace("�", "'").strip()


def expects_answer(agent_text):
    """True when the agent's words end on something the caller should answer (a question or a goodbye).
    Holding lines such as "Sure, I'll find what's open." return False: a tool is running and more is coming."""
    tail = re.split(r"(?<=[.!?])\s+", agent_text.strip())[-2:]
    return any(s.rstrip().endswith("?") for s in tail) or bool(
        re.search(r"\b(goodbye|bye|have a (great|good|nice) day)\b", agent_text, re.I))


def appointment_id_from(record):
    """Booking confirmation number the agent read out, e.g. 'APT-RXVRXM' (None if the call never got there)."""
    for t in (record or {}).get("transcript") or []:
        if t.get("role") == "assistant":
            m = re.search(r"confirmation (?:number|code|id) is ([A-Z0-9][A-Z0-9-]{3,})", t.get("content") or "", re.I)
            if m:
                return m.group(1).rstrip(".")
    return None


async def run_one_dynamic_user(session, base, auth_headers, agent_id, patient, ai_config,
                               greeting_timeout, reply_timeout, livekit_url_override=None, audio_path=None,
                               settle_time=1.5, llm_timeout=10, llm_input="text", hold_timeout=20, tag="",
                               noise_type="none", noise_level=0.0):
    r = {
        "started_at": datetime.now().isoformat(),
        "success": False,
        "steps": {},
        "patient": patient,
        "noise": {"type": noise_type, "level": noise_level} if noise_type != "none" else None,
    }
    t0 = time.monotonic()

    # 1. create the web call
    async with session.post(f"{base}/v1/agents/{agent_id}/web-calls", json={}, headers=auth_headers) as resp:
        body = await resp.json(content_type=None)
        r["steps"]["web_call_http_status"] = resp.status
        r["steps"]["web_call_ms"] = round((time.monotonic() - t0) * 1000)
        r["web_call_response"] = redact(body)
        if resp.status >= 300:
            r["error"] = f"web-calls returned HTTP {resp.status}"
            return r

    token = find_key(body, ["token", "join_token", "participant_token", "access_token"])
    url = find_key(body, ["livekit_url", "ws_url", "server_url", "url"])
    call_id = find_key(body, ["call_id", "id"])
    r["call_id"] = call_id
    r["livekit_url_returned"] = url
    if livekit_url_override:
        url = livekit_url_override
    r["livekit_url_used"] = url
    if not token or not url:
        r["error"] = "could not find join token / livekit_url in web-calls response"
        return r

    room = rtc.Room()
    listener = AgentListener()
    mic = Mic(noise_type=noise_type, noise_level=noise_level)
    tasks = []
    events = r["events"] = []
    t_join = time.monotonic()

    def log(name, **kw):
        events.append({"t_ms": round((time.monotonic() - t_join) * 1000), "event": name, **kw})

    @room.on("track_subscribed")
    def _on_track(track, publication, participant):
        log("track_subscribed", participant=participant.identity, kind=str(track.kind))
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            r["agent_identity"] = participant.identity
            tasks.append(asyncio.create_task(listener.consume(track)))

    room.on("participant_connected", lambda p: log("participant_connected", participant=p.identity))
    room.on("participant_disconnected", lambda p: log("participant_disconnected", participant=p.identity))
    room.on("track_published", lambda pub, p: log("track_published", participant=p.identity, kind=str(pub.kind)))
    room.on("track_subscription_failed", lambda p, sid, err: log("track_subscription_failed", participant=p.identity, error=str(err)))
    room.on("connection_state_changed", lambda s: log("connection_state", state=str(s)))
    room.on("disconnected", lambda reason: (log("disconnected", reason=str(reason)),
                                           setattr(listener, "closed", True)))
    watch_transcriptions(room, log, tasks)

    temp_dir = Path("uploads/temp_audio")
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        t_join = time.monotonic()
        await room.connect(url, token)
        r["steps"]["room_join_ms"] = round((time.monotonic() - t_join) * 1000)
        r["room"] = room.name
        log("connected", already_in_room=[p.identity for p in room.remote_participants.values()])

        track = rtc.LocalAudioTrack.create_audio_track("mic", mic.source)
        mic_pub = await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        log("mic_published", sid=mic_pub.sid)
        tasks.append(asyncio.create_task(mic.run()))

        first_name = patient.get("first_name", "John")
        last_name = patient.get("last_name", "Smith")
        dob = patient.get("dob", "May 20th, 1985")
        phone = patient.get("phone", "9876543210")
        specialty = patient.get("specialty", "Cardiology")

        system_prompt = (
            f"You are a patient calling a medical clinic over the phone to book an appointment.\n"
            f"Your personal details:\n"
            f"- Full Name: {first_name} {last_name}\n"
            f"- First Name: {first_name}\n"
            f"- Last Name: {last_name}\n"
            f"- Date of Birth: {dob}\n"
            f"- Phone Number: {phone}\n"
            f"- Desired Specialty: {specialty}\n"
            f"- Date/Time Preference: {patient.get('time_preference', 'next Monday morning')}\n\n"
            f"Rules:\n"
            f"1. Be extremely concise (1 short sentence, maximum 10-12 words). Speak naturally as a caller.\n"
            f"2. Answer directly what the receptionist asks. If asked for full name, give both. If asked for first name only, give first name.\n"
            f"3. If asked for phone number, state: {phone}.\n"
            f"4. If asked to confirm (e.g. spelling or date), say: 'Yes, that is correct.'\n"
            f"5. If offered appointment times, say: 'The first option, please.'\n"
            f"6. If asked if you need anything else or call wraps up, say: 'No, that's all. Thank you.'"
        )

        seen_agent_msgs = [0]  # how many agent transcript lines were already handed to the caller LLM
        gemini_history = []
        openai_messages = [{"role": "system", "content": system_prompt}]
        r["turns"] = []
        turn_latencies = []

        # 2. Greeting
        first_utt, last_end = await listener.wait_for_full_reply(t_join, greeting_timeout, settle_time=settle_time)
        t_detected = time.monotonic()
        if not first_utt:
            r["error"] = "silent call: agent never spoke (no greeting heard)"
            return r

        r["steps"]["greeting_start_ms"] = round((first_utt[0] - t_join) * 1000)
        r["steps"]["call_setup_ms"] = round((first_utt[0] - t0) * 1000)
        r["steps"]["greeting_duration_ms"] = round((last_end - first_utt[0]) * 1000)

        # Dynamic loop (up to 15 turns)
        for turn_idx in range(1, 16):
            if room.connection_state == rtc.ConnectionState.CONN_DISCONNECTED:
                log("call_disconnected_by_agent", turn=turn_idx)
                break

            # 1. Generate caller response via Gemini or OpenAI
            llm_info = {}
            # Prefer the agent's words as text (from the wrapper transcript): small, fast LLM requests.
            # Fall back to sending the agent's audio when the text is not there yet.
            agent_text = None
            hold_ms = 0
            if llm_input == "text":
                agent_text = await fetch_new_agent_text(session, base, auth_headers, call_id, seen_agent_msgs)
                # The agent often says a holding line ("Sure, I'll find what's open.") and goes quiet while a
                # tool runs. Answering that talks over the real question, so keep listening until it asks one.
                t_hold = time.monotonic()
                while agent_text and not expects_answer(agent_text) and time.monotonic() - t_hold < hold_timeout:
                    more_utt, more_end = await listener.wait_for_full_reply(
                        last_end, hold_timeout - (time.monotonic() - t_hold), settle_time=settle_time)
                    if not more_utt:
                        break
                    last_end, t_detected = more_end, time.monotonic()
                    more = None
                    for _ in range(4):  # the wrapper transcript can lag the audio by a moment
                        more = await fetch_new_agent_text(session, base, auth_headers, call_id, seen_agent_msgs)
                        if more:
                            break
                        await asyncio.sleep(0.5)
                    if more:
                        agent_text = f"{agent_text} {more}"
                hold_ms = round((time.monotonic() - t_hold) * 1000)
                if room.connection_state == rtc.ConnectionState.CONN_DISCONNECTED:
                    log("call_disconnected_by_agent", turn=turn_idx)
                    break
            agent_source = "transcript" if agent_text else "audio"
            t_llm = time.monotonic()  # after any hold, so detect + llm + tts add up to caller_gap
            if not agent_text:
                agent_wav = temp_dir / f"agent_turn_{turn_idx}_{time.time_ns()}.wav"
                extract_agent_audio(listener.tracks, first_utt[0], last_end, agent_wav)
                audio_b64 = base64.b64encode(agent_wav.read_bytes()).decode("utf-8")

            if ai_config["provider"] == "gemini":
                if agent_text:
                    parts = [{"text": f'The clinic receptionist just said: "{agent_text}" '
                                      "Answer directly in 1 short spoken sentence."}]
                else:
                    parts = [{"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
                             {"text": "The clinic receptionist just said this to you on the phone. "
                                      "Answer directly in 1 short spoken sentence."}]
                gemini_history.append({"role": "user", "parts": parts})
                caller_text = await gemini_generate_reply(
                    session,
                    ai_config["api_key"],
                    ai_config["model"],
                    ai_config["fallbacks"],
                    gemini_history,
                    system_prompt,
                    info=llm_info,
                    request_timeout=llm_timeout,
                )
                gemini_history.append({"role": "model", "parts": [{"text": caller_text}]})
            else:
                if agent_text:
                    agent_reply_text = agent_text
                else:
                    with open(agent_wav, "rb") as f:
                        t_res = await ai_config["client"].audio.transcriptions.create(model="whisper-1", file=f)
                    agent_reply_text = t_res.text.strip()
                openai_messages.append({"role": "user", "content": agent_reply_text})
                chat_res = await ai_config["client"].chat.completions.create(
                    model="gpt-4o-mini",
                    messages=openai_messages,
                    max_tokens=50,
                    temperature=0.2,
                )
                caller_text = chat_res.choices[0].message.content.strip()
                openai_messages.append({"role": "assistant", "content": caller_text})

            t_tts = time.monotonic()
            user_wav = temp_dir / f"caller_turn_{turn_idx}_{time.time_ns()}.wav"
            await asyncio.to_thread(text_to_wav, caller_text, user_wav)
            samples = load_wav(user_wav)
            t_speak = time.monotonic()
            # caller-side delay = agent stopped talking -> caller starts talking
            timing = {
                "detect_wait_ms": round((t_llm - last_end) * 1000),  # sure agent finished + transcript fetch
                "llm_ms": round((t_tts - t_llm) * 1000),
                "tts_ms": round((t_speak - t_tts) * 1000),
                "caller_gap_ms": round((t_speak - last_end) * 1000),
            }
            turn_info = {
                "turn": turn_idx,
                "caller_spoken": caller_text,
                "question_ms": round(len(samples) / SAMPLE_RATE * 1000),
                "caller_timing": timing,
                "agent_text_source": agent_source,
                "agent_text": agent_text,
            }
            if hold_ms > 1000:
                turn_info["agent_hold_ms"] = hold_ms  # waited this long for the agent to finish (tool running)
            if llm_info:
                turn_info["llm_model"] = llm_info.get("model")
                if llm_info.get("attempts"):
                    turn_info["llm_failed_attempts"] = llm_info["attempts"]
            slow = "".join(f" [{a['model']}: {a['status']} after {a['ms']} ms]" for a in llm_info.get("attempts", []))
            print(f"  {tag}[Caller Turn {turn_idx:>2}]: {caller_text}")
            print(f"      {tag}caller delay {timing['caller_gap_ms']/1000:.1f}s = wait {timing['detect_wait_ms']/1000:.1f}s"
                  f" + LLM {timing['llm_ms']/1000:.1f}s ({agent_source}) + speech {timing['tts_ms']/1000:.1f}s{slow}")

            log("turn_user_speak_start", turn=turn_idx, text=caller_text)
            mic.say(samples)
            await mic.speech_done.wait()
            t_spoken = time.monotonic()
            log("turn_user_speak_end", turn=turn_idx, trailing_silence_ms=trailing_silence_ms(samples))

            first_utt, last_end = await listener.wait_for_full_reply(t_spoken, reply_timeout, settle_time=settle_time)
            t_detected = time.monotonic()
            if not first_utt:
                if re.search(r"\b(thank you|no|bye)\b", caller_text, re.I):
                    log("call_ended_cleanly", turn=turn_idx)
                    r["turns"].append(turn_info)
                    break
                turn_info["error"] = f"agent did not reply within {reply_timeout}s"
                r["turns"].append(turn_info)
                r["error"] = f"turn {turn_idx}: agent did not reply within {reply_timeout}s"
                break

            agent_resp_ms = round((first_utt[0] - t_spoken) * 1000)
            reply_dur_ms = round((last_end - first_utt[0]) * 1000)
            turn_info["agent_response_ms"] = agent_resp_ms
            turn_info["reply_duration_ms"] = reply_dur_ms
            turn_latencies.append(agent_resp_ms)
            log("turn_agent_reply_received", turn=turn_idx, latency_ms=agent_resp_ms)
            r["turns"].append(turn_info)

            # Check if wrap up
            if any(w in caller_text.lower() for w in ["thank you", "bye", "that's all"]) and turn_idx >= 5:
                log("call_completed", turn=turn_idx)
                break

        gaps = [t["caller_timing"]["caller_gap_ms"] for t in r["turns"] if "caller_timing" in t]
        if gaps:
            r["steps"]["avg_caller_gap_ms"] = round(sum(gaps) / len(gaps))
        if turn_latencies:
            r["steps"]["agent_response_ms"] = turn_latencies[0]
            r["steps"]["avg_agent_response_ms"] = round(sum(turn_latencies) / len(turn_latencies))
        r["success"] = True
        return r
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
        return r
    finally:
        mic.stop()
        for t in tasks:
            t.cancel()
        await room.disconnect()
        if audio_path and (mic.recorder.chunks or any(x.chunks for x in listener.recorders)):
            write_stereo(audio_path, t_join, mic.recorder, listener.recorders)
            r["audio_file"] = str(audio_path)
        r["steps"]["total_ms"] = round((time.monotonic() - t0) * 1000)


async def main():
    p = argparse.ArgumentParser(description="AI Labs web-call test")
    p.add_argument("--dynamic", action="store_true",
                   help="Use dynamic AI caller (Gemini or OpenAI)")
    p.add_argument("--gemini-api-key", default=None,
                   help="Gemini API key (falls back to GEMINI_API_KEY in .env)")
    p.add_argument("--openai-api-key", default=None,
                   help="OpenAI API key (falls back to OPENAI_API_KEY in .env)")
    p.add_argument("--patients-file", default="patients.json",
                   help="JSON file containing list of patient profiles for dynamic mode (default: patients.json)")
    p.add_argument("--patient-count", type=int, default=5,
                   help="Number of patients to run in dynamic mode (default: 5)")
    p.add_argument("--scenario", choices=list(PRESET_SCENARIOS.keys()), default=None,
                   help="Run a predefined multi-turn scenario (e.g. 'booking')")
    p.add_argument("--first-name", default="Anna", help="Caller first name for booking scenario (default: Anna)")
    p.add_argument("--last-name", default="Beckett", help="Caller last name for booking scenario (default: Beckett)")
    p.add_argument("--dob", default="October 4th, 1973", help="Caller date of birth for booking scenario (default: October 4th, 1973)")
    p.add_argument("--phone", default="9876543210", help="Caller phone number (default: 9876543210)")
    p.add_argument("--specialty", default="primary care", help="Specialty for booking scenario (default: primary care)")
    p.add_argument("--turns-file", default=None, help="JSON or text file containing a list of turn utterances")
    p.add_argument("--question-text", default=None,
                   help='single question text, e.g. "I want to book an appointment"')
    p.add_argument("--question", default=None,
                   help="or: a recorded question WAV (16 kHz mono 16-bit) instead of text")
    p.add_argument("--greeting-timeout", type=float, default=30)
    p.add_argument("--reply-timeout", type=float, default=30)
    p.add_argument("--output", default=None, help="result JSON path")
    p.add_argument("--livekit-url", default=None,  # falls back to LIVEKIT_URL in .env
                   help="join this LiveKit URL instead of the one the wrapper returns")
    p.add_argument("--runs", type=int, default=1, help="number of calls, one after another")
    p.add_argument("--pause", type=float, default=5, help="seconds between calls")
    p.add_argument("--end-of-speech", type=float, default=END_OF_SPEECH_S,
                   help=f"seconds of silence that end an agent utterance (default {END_OF_SPEECH_S})")
    p.add_argument("--settle", type=float, default=1.5,
                   help="dynamic mode: extra seconds to wait in case the agent continues talking (default 1.5)")
    p.add_argument("--llm-input", choices=["text", "audio"], default="text",
                   help="dynamic mode: give the caller LLM the agent's words as transcript text (fast, default) "
                        "or as audio (old behaviour)")
    p.add_argument("--llm-timeout", type=float, default=10,
                   help="dynamic mode: max seconds per Gemini request before trying the next model (default 10)")
    p.add_argument("--hold-timeout", type=float, default=20,
                   help="dynamic mode: when the agent's words are not a question (e.g. 'Sure, I'll find what's "
                        "open.'), keep listening up to this many seconds before answering (default 20)")
    p.add_argument("--concurrent", type=int, default=1,
                   help="dynamic mode: number of calls running at the same time (default 1 = one after another)")
    p.add_argument("--stagger", type=float, default=1.0,
                   help="dynamic mode with --concurrent: seconds between call starts (default 1)")
    p.add_argument("--no-latency-pdf", action="store_true",
                   help="skip the per-call latency PDF (default: written next to the result JSON)")
    p.add_argument("--noise-type", choices=["none", "low", "medium", "high", "fan", "typing", "cafe", "hum", "babble"],
                   default="none",
                   help="Inject background noise into caller microphone to test agent noise cancellation: "
                        "'low' (fan/hum drone), 'medium' (keyboard typing clicks), 'high' (cafe chatter/babble)")
    p.add_argument("--noise-level", type=float, default=None,
                   help="Background noise level from 0.0 to 1.0 (defaults: 0.15 for low, 0.25 for medium, 0.40 for high)")
    args = p.parse_args()
    globals()["END_OF_SPEECH_S"] = args.end_of_speech

    if args.noise_level is None:
        noise_defaults = {
            "low": 0.15, "fan": 0.15, "hum": 0.15,
            "medium": 0.25, "typing": 0.25, "clicks": 0.25,
            "high": 0.40, "cafe": 0.40, "babble": 0.40,
        }
        args.noise_level = noise_defaults.get(args.noise_type, 0.0)

    load_env()
    if not args.livekit_url and os.environ.get("LIVEKIT_URL"):
        args.livekit_url = os.environ["LIVEKIT_URL"]
    base = os.environ.get("WRAPPER_BASE_URL", "").rstrip("/")
    user = os.environ.get("WRAPPER_USERNAME")
    pwd = os.environ.get("WRAPPER_PASSWORD")
    agent_id = os.environ.get("AGENT_ID")
    missing = [n for n, v in [("WRAPPER_BASE_URL", base), ("WRAPPER_USERNAME", user),
                              ("WRAPPER_PASSWORD", pwd), ("AGENT_ID", agent_id)] if not v]
    if missing:
        print(f"Missing in .env: {', '.join(missing)}")
        return 1

    timeout = aiohttp.ClientTimeout(total=30)

    # DYNAMIC AI CALLER MODE (GEMINI OR OPENAI)
    if args.dynamic:
        gemini_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")

        if gemini_key:
            model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
            fallbacks = os.environ.get("GEMINI_MODEL_FALLBACKS", "gemini-3.8-flash,gemini-3.1-flash-lite")
            ai_config = {"provider": "gemini", "api_key": gemini_key, "model": model, "fallbacks": fallbacks}
            print(f"Using Dynamic Caller Engine: Gemini ({model})")
        elif openai_key:
            if AsyncOpenAI is None:
                print("Error: openai package is not installed. Run: pip install openai")
                return 1
            ai_config = {"provider": "openai", "client": AsyncOpenAI(api_key=openai_key)}
            print("Using Dynamic Caller Engine: OpenAI (gpt-4o-mini + Whisper)")
        else:
            print("Error: Dynamic mode requires GEMINI_API_KEY or OPENAI_API_KEY in .env")
            return 1

        p_path = Path(args.patients_file)
        if not p_path.exists():
            print(f"Error: Patients file not found: {p_path}")
            return 1
        pool = json.loads(p_path.read_text())
        patients = [pool[k % len(pool)] for k in range(args.patient_count)]  # repeats patients if count > file
        print(f"Loaded {len(patients)} patient profile(s) from {p_path} for dynamic testing.")

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{base}/v1/auth/login", data={"username": user, "password": pwd},
                                    headers=host_header()) as resp:
                body = await resp.json(content_type=None)
                if resp.status >= 300:
                    print(f"Login failed: HTTP {resp.status} {redact(body)}")
                    return 1
            api_token = find_key(body, ["access_token", "token", "jwt"])
            if not api_token:
                print(f"Login OK but no token found in response keys: {list(body)}")
                return 1
            headers = {"Authorization": f"Bearer {api_token}", **host_header()}
            print(f"Logged in. Calling agent {agent_id} dynamically for {len(patients)} patient(s) ...")

            concurrent = max(1, args.concurrent)
            if concurrent > 1:
                print(f"Running {len(patients)} call(s), up to {concurrent} at the same time, "
                      f"starting {args.stagger:g} s apart.")
            gate = asyncio.Semaphore(concurrent)
            finished = []  # (call_id, record, bot turns, result) for the combined summary

            async def run_patient(i, patient):
                if concurrent == 1 and i > 1:
                    await asyncio.sleep(args.pause)
                elif concurrent > 1:
                    await asyncio.sleep((i - 1) * args.stagger)
                async with gate:
                    return await one_patient(i, patient)

            async def one_patient(i, patient):
                tag = f"[P{i}] " if concurrent > 1 else ""
                out = (args.output if args.output and len(patients) == 1
                       else f"results/dynamic_patient_{i}_{datetime.now():%Y%m%d_%H%M%S}.json")
                Path(out).parent.mkdir(exist_ok=True)
                print(f"\n==================================================")
                print(f"{tag}Patient {i}/{len(patients)}: {patient.get('first_name')} {patient.get('last_name')} | DOB: {patient.get('dob')} | Specialty: {patient.get('specialty')}")
                result = await run_one_dynamic_user(session, base, headers, agent_id, patient, ai_config,
                                                    args.greeting_timeout, args.reply_timeout, args.livekit_url,
                                                    audio_path=Path(out).with_suffix(".wav"),
                                                    settle_time=args.settle, llm_timeout=args.llm_timeout,
                                                    llm_input=args.llm_input, hold_timeout=args.hold_timeout,
                                                    tag=tag,
                                                    noise_type=args.noise_type, noise_level=args.noise_level)
                record = None
                if result.get("call_id"):
                    await asyncio.sleep(3)
                    result["call_record_http_status"], record = await fetch_call_record(
                        session, base, headers, result["call_id"])
                    result["call_record"] = redact(record)
                    result["appointment_id"] = appointment_id_from(record)
                result["mode"] = f"Dynamic AI Caller ({ai_config['provider'].title()})"
                add_bot_stt(result)
                Path(out).write_text(json.dumps(result, indent=2, default=str))
                print_result(result, out, i, len(patients))
                if record and result["call_record_http_status"] < 300:
                    finished.append((result["call_id"], record, result.get("turns"), result))
                    if not args.no_latency_pdf:
                        write_latency_pdf(result["call_id"], record, out, result.get("turns"))
                return result

            results = await asyncio.gather(*(run_patient(i, pt) for i, pt in enumerate(patients, 1)))
            passed = sum(bool(r.get("success")) for r in results)
            if concurrent > 1 and finished:
                summarise_parallel(finished, concurrent, args.no_latency_pdf)

            print(f"\nDynamic Testing Complete: {passed}/{len(patients)} calls passed.")
            return 0 if passed == len(patients) else 2

    # PRE-SCRIPTED / SCENARIO / SINGLE QUESTION MODE
    turns_texts = []
    if args.scenario:
        template = PRESET_SCENARIOS[args.scenario]
        turns_texts = [
            t.format(
                first_name=args.first_name,
                last_name=args.last_name,
                dob=args.dob,
                phone=args.phone,
                specialty=args.specialty,
            )
            for t in template
        ]
    elif args.turns_file:
        p_file = Path(args.turns_file)
        if p_file.suffix.lower() == ".json":
            turns_texts = json.loads(p_file.read_text())
        else:
            turns_texts = [line.strip() for line in p_file.read_text().splitlines() if line.strip()]
    elif args.question:
        turns = [(f"WAV: {args.question}", load_wav(args.question))]
    else:
        turns_texts = [args.question_text or DEFAULT_QUESTION_TEXT]

    if not args.question:
        cache_dir = Path("uploads/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        turns = []
        for text in turns_texts:
            h = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
            wav_path = cache_dir / f"turn_{h}.wav"
            if not wav_path.exists():
                text_to_wav(text, str(wav_path))
            turns.append((text, load_wav(wav_path)))

    mode_name = f"Scenario '{args.scenario}'" if args.scenario else ("Single Question" if len(turns) == 1 else f"Multi-turn ({len(turns)} turns)")
    print(f"Conversation Mode: {mode_name}")
    for idx, (t_text, _) in enumerate(turns, start=1):
        print(f"  Turn {idx:>2}: {t_text}")

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{base}/v1/auth/login", data={"username": user, "password": pwd},
                                    headers=host_header()) as resp:
            body = await resp.json(content_type=None)
            if resp.status >= 300:
                print(f"Login failed: HTTP {resp.status} {redact(body)}")
                return 1
        api_token = find_key(body, ["access_token", "token", "jwt"])
        if not api_token:
            print(f"Login OK but no token found in response keys: {list(body)}")
            return 1
        headers = {"Authorization": f"Bearer {api_token}", **host_header()}
        print(f"Logged in. Calling agent {agent_id} ({args.runs} call(s), one at a time) ...")

        passed = 0
        for i in range(1, args.runs + 1):
            if i > 1:
                await asyncio.sleep(args.pause)
            out = (args.output if args.output and args.runs == 1
                   else f"results/webcall_1user_{datetime.now():%Y%m%d_%H%M%S}.json")
            Path(out).parent.mkdir(exist_ok=True)
            result = await run_one_user(session, base, headers, agent_id, turns,
                                        args.greeting_timeout, args.reply_timeout, args.livekit_url,
                                        audio_path=Path(out).with_suffix(".wav"),
                                        noise_type=args.noise_type, noise_level=args.noise_level)
            record = None
            if result.get("call_id"):
                await asyncio.sleep(3)  # let the wrapper store the call record
                result["call_record_http_status"], record = await fetch_call_record(
                    session, base, headers, result["call_id"])
                result["call_record"] = redact(record)
                result["appointment_id"] = appointment_id_from(record)
            result["mode"] = mode_name
            result["turns_summary"] = [
                {
                    "turn": t["turn"],
                    "text": t.get("text") or t.get("caller_spoken"),
                    "question_ms": t.get("question_ms"),
                    "agent_response_ms": t.get("agent_response_ms"),
                    "reply_duration_ms": t.get("reply_duration_ms"),
                    "error": t.get("error"),
                }
                for t in result.get("turns", [])
            ]
            add_bot_stt(result)
            Path(out).write_text(json.dumps(result, indent=2, default=str))
            passed += result["success"]
            print_result(result, out, i, args.runs)
            if record and result["call_record_http_status"] < 300 and not args.no_latency_pdf:
                write_latency_pdf(result["call_id"], record, out)

    if args.runs > 1:
        print(f"\nDone: {passed}/{args.runs} calls passed. Build the report with: python make_report.py")
    return 0 if passed == args.runs else 2


def summarise_parallel(finished, concurrent, no_pdf):
    """Real wait across all calls that ran in parallel, plus one combined latency PDF."""
    import call_latency as cl
    report, all_turns = [], []
    for cid, record, bot_turns, _ in finished:
        res = cl.analyse(record)
        if res is not None:
            res["caller_turns"] = bot_turns
            all_turns += [(cid, i, t) for i, t in enumerate(res["turns"], 1)]
        report.append((cid, res, None))
    print("\n" + "=" * 50 + f"\nPARALLEL RUN: {len(finished)} call(s), up to {concurrent} at once")
    for cid, _, _, result in finished:
        print(f"  {cid}  appointment {result.get('appointment_id') or 'NONE'}")
    waits = [t["wait"] for _, _, t in all_turns]
    if waits:
        print(f"  real wait over {len(waits)} turns: median {statistics.median(waits):.0f} ms, "
              f"P90 {cl.pct(waits, 90):.0f} ms, {sum(w > 3000 for w in waits)} above 3 s, "
              f"{sum(w <= 2500 for w in waits)} within 2.5 s")
    if not no_pdf and all_turns:
        path = f"results/parallel_{concurrent}x_{datetime.now():%Y%m%d_%H%M%S}_latency.pdf"
        try:
            cl.write_pdf(path, report, all_turns)
            print(f"Combined latency PDF: {path}")
        except Exception as e:
            print(f"  combined PDF failed: {e}")


def print_result(result, out, i, n):
    print("\n" + "=" * 50 + (f"  call {i}/{n}" if n > 1 else ""))
    print("RESULT:", "SUCCESS" if result["success"] else f"FAILED - {result.get('error')}")
    for k, v in result["steps"].items():
        print(f"  {k:<24} {v}")
    print(f"  call_id                  {result.get('call_id')}")
    if result.get("noise"):
        print(f"  noise_injection          {result['noise']['type']} (level {result['noise']['level']})")
    if "appointment_id" in result:
        print(f"  appointment_id           {result['appointment_id'] or 'NONE - booking was not confirmed'}")
    if result.get("patient"):
        p = result["patient"]
        print(f"  patient                  {p.get('first_name')} {p.get('last_name')} (DOB: {p.get('dob')})")
    if result.get("turns"):
        print("\n  Turn Details:")
        for t in result["turns"]:
            resp = t.get('agent_response_ms', 'N/A')
            dur = t.get('reply_duration_ms', 'N/A')
            txt = t.get('text') or t.get('caller_spoken') or ''
            print(f"    Turn {t['turn']:>2} [{txt[:38]:<38}] -> resp: {str(resp):>5} ms, dur: {str(dur):>5} ms")
    for ev in result.get("events", []):
        print(f"  [{ev['t_ms']:>6} ms] " + ", ".join(f"{k}={v}" for k, v in ev.items() if k != "t_ms"))
    print(f"Saved: {out}")
    if result.get("audio_file"):
        print(f"Audio: {result['audio_file']}  (left = caller question, right = agent)")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
