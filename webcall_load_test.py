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

Usage:
    python webcall_load_test.py --question uploads/question.wav
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import subprocess
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


def text_to_wav(text, path):
    """Speak text with the Windows built-in voice into a 16 kHz mono 16-bit WAV."""
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
        while time.monotonic() < deadline:
            for tl in self.tracks:
                for u in tl.utterances:
                    if u[0] >= after:
                        return u
            await asyncio.sleep(0.02)
        return None

    async def wait_for_utterance_end(self, utt, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and utt[1] is None:
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


class Mic:
    """Publishes a continuous mic track: silence, or speech when asked."""

    def __init__(self):
        self.source = rtc.AudioSource(SAMPLE_RATE, 1)
        self._queue = None
        self._running = True
        self.speech_done = asyncio.Event()
        self.recorder = Recorder()

    def say(self, samples):
        self.speech_done.clear()
        self._queue = samples

    async def run(self):
        silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        next_t = time.monotonic()
        pos = 0
        while self._running:
            if self._queue is not None:
                chunk = self._queue[pos:pos + FRAME_SAMPLES]
                pos += FRAME_SAMPLES
                if chunk.size < FRAME_SAMPLES:
                    chunk = np.pad(chunk, (0, FRAME_SAMPLES - chunk.size))
                if pos >= self._queue.size:
                    self._queue, pos = None, 0
                    self.speech_done.set()
            else:
                chunk = silence
            frame = rtc.AudioFrame.create(SAMPLE_RATE, 1, FRAME_SAMPLES)
            np.copyto(np.frombuffer(frame.data, dtype=np.int16), chunk)
            await self.source.capture_frame(frame)
            self.recorder.add(chunk)
            next_t += FRAME_MS / 1000
            await asyncio.sleep(max(0, next_t - time.monotonic()))

    def stop(self):
        self._running = False


async def run_one_user(session, base, auth_headers, agent_id, turns, greeting_timeout, reply_timeout,
                       livekit_url_override=None, audio_path=None):
    r = {"started_at": datetime.now().isoformat(), "success": False, "steps": {}}
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
    mic = Mic()
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
    room.on("disconnected", lambda reason: log("disconnected", reason=str(reason)))

    try:
        t_join = time.monotonic()
        await room.connect(url, token)
        r["steps"]["room_join_ms"] = round((time.monotonic() - t_join) * 1000)
        r["room"] = room.name
        log("connected", already_in_room=[p.identity for p in room.remote_participants.values()])

        track = rtc.LocalAudioTrack.create_audio_track("mic", mic.source)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
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
            log("turn_user_speak_end", turn=turn_idx)

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


async def run_one_dynamic_user(session, base, auth_headers, agent_id, patient, ai_config,
                               greeting_timeout, reply_timeout, livekit_url_override=None, audio_path=None,
                               settle_time=1.5, llm_timeout=10, llm_input="text"):
    r = {"started_at": datetime.now().isoformat(), "success": False, "steps": {}, "patient": patient}
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
    mic = Mic()
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
    room.on("disconnected", lambda reason: log("disconnected", reason=str(reason)))

    temp_dir = Path("uploads/temp_audio")
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        t_join = time.monotonic()
        await room.connect(url, token)
        r["steps"]["room_join_ms"] = round((time.monotonic() - t_join) * 1000)
        r["room"] = room.name
        log("connected", already_in_room=[p.identity for p in room.remote_participants.values()])

        track = rtc.LocalAudioTrack.create_audio_track("mic", mic.source)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
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
            f"- Date/Time Preference: next Monday morning\n\n"
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
            t_llm = time.monotonic()
            llm_info = {}
            # Prefer the agent's words as text (from the wrapper transcript): small, fast LLM requests.
            # Fall back to sending the agent's audio when the text is not there yet.
            agent_text = None
            if llm_input == "text":
                agent_text = await fetch_new_agent_text(session, base, auth_headers, call_id, seen_agent_msgs)
            agent_source = "transcript" if agent_text else "audio"
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
                "detect_wait_ms": round((t_detected - last_end) * 1000),
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
            }
            if llm_info:
                turn_info["llm_model"] = llm_info.get("model")
                if llm_info.get("attempts"):
                    turn_info["llm_failed_attempts"] = llm_info["attempts"]
            slow = "".join(f" [{a['model']}: {a['status']} after {a['ms']} ms]" for a in llm_info.get("attempts", []))
            print(f"  [Caller Turn {turn_idx:>2}]: {caller_text}")
            print(f"      caller delay {timing['caller_gap_ms']/1000:.1f}s = wait {timing['detect_wait_ms']/1000:.1f}s"
                  f" + LLM {timing['llm_ms']/1000:.1f}s ({agent_source}) + speech {timing['tts_ms']/1000:.1f}s{slow}")

            log("turn_user_speak_start", turn=turn_idx, text=caller_text)
            mic.say(samples)
            await mic.speech_done.wait()
            t_spoken = time.monotonic()
            log("turn_user_speak_end", turn=turn_idx)

            first_utt, last_end = await listener.wait_for_full_reply(t_spoken, reply_timeout, settle_time=settle_time)
            t_detected = time.monotonic()
            if not first_utt:
                if "thank you" in caller_text.lower() or "no" in caller_text.lower():
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
    args = p.parse_args()
    globals()["END_OF_SPEECH_S"] = args.end_of_speech

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
        patients = json.loads(p_path.read_text())[:args.patient_count]
        print(f"Loaded {len(patients)} patient profile(s) from {p_path} for dynamic testing.")

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{base}/v1/auth/login", data={"username": user, "password": pwd}) as resp:
                body = await resp.json(content_type=None)
                if resp.status >= 300:
                    print(f"Login failed: HTTP {resp.status} {redact(body)}")
                    return 1
            api_token = find_key(body, ["access_token", "token", "jwt"])
            if not api_token:
                print(f"Login OK but no token found in response keys: {list(body)}")
                return 1
            headers = {"Authorization": f"Bearer {api_token}"}
            print(f"Logged in. Calling agent {agent_id} dynamically for {len(patients)} patient(s) ...")

            passed = 0
            for i, patient in enumerate(patients, 1):
                if i > 1:
                    await asyncio.sleep(args.pause)
                out = (args.output if args.output and len(patients) == 1
                       else f"results/dynamic_patient_{i}_{datetime.now():%Y%m%d_%H%M%S}.json")
                Path(out).parent.mkdir(exist_ok=True)
                print(f"\n==================================================")
                print(f"Patient {i}/{len(patients)}: {patient.get('first_name')} {patient.get('last_name')} | DOB: {patient.get('dob')} | Specialty: {patient.get('specialty')}")
                result = await run_one_dynamic_user(session, base, headers, agent_id, patient, ai_config,
                                                    args.greeting_timeout, args.reply_timeout, args.livekit_url,
                                                    audio_path=Path(out).with_suffix(".wav"),
                                                    settle_time=args.settle, llm_timeout=args.llm_timeout,
                                                    llm_input=args.llm_input)
                if result.get("call_id"):
                    await asyncio.sleep(3)
                    async with session.get(f"{base}/v1/calls/{result['call_id']}", headers=headers) as resp:
                        result["call_record_http_status"] = resp.status
                        result["call_record"] = redact(await resp.json(content_type=None))
                result["mode"] = f"Dynamic AI Caller ({ai_config['provider'].title()})"
                Path(out).write_text(json.dumps(result, indent=2, default=str))
                passed += result["success"]
                print_result(result, out, i, len(patients))

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
        async with session.post(f"{base}/v1/auth/login", data={"username": user, "password": pwd}) as resp:
            body = await resp.json(content_type=None)
            if resp.status >= 300:
                print(f"Login failed: HTTP {resp.status} {redact(body)}")
                return 1
        api_token = find_key(body, ["access_token", "token", "jwt"])
        if not api_token:
            print(f"Login OK but no token found in response keys: {list(body)}")
            return 1
        headers = {"Authorization": f"Bearer {api_token}"}
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
                                        audio_path=Path(out).with_suffix(".wav"))
            if result.get("call_id"):
                await asyncio.sleep(3)  # let the wrapper store the call record
                async with session.get(f"{base}/v1/calls/{result['call_id']}", headers=headers) as resp:
                    result["call_record_http_status"] = resp.status
                    result["call_record"] = redact(await resp.json(content_type=None))
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
            Path(out).write_text(json.dumps(result, indent=2, default=str))
            passed += result["success"]
            print_result(result, out, i, args.runs)

    if args.runs > 1:
        print(f"\nDone: {passed}/{args.runs} calls passed. Build the report with: python make_report.py")
    return 0 if passed == args.runs else 2


def print_result(result, out, i, n):
    print("\n" + "=" * 50 + (f"  call {i}/{n}" if n > 1 else ""))
    print("RESULT:", "SUCCESS" if result["success"] else f"FAILED - {result.get('error')}")
    for k, v in result["steps"].items():
        print(f"  {k:<24} {v}")
    print(f"  call_id                  {result.get('call_id')}")
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
