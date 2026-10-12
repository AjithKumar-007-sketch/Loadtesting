#!/usr/bin/env python3
"""
Simple Livekit Audio Test Script with FastAPI Server

This script creates a Livekit call and plays audio from a WAV file
instead of using the complex websocket infrastructure from the main codebase.

Usage:
    CLI: python simple_livekit_audio_test.py --to-phone "+1234567890" --wav-file "test_audio.wav"
    API Server: python simple_livekit_audio_test.py --server
    API Server with custom port: python simple_livekit_audio_test.py --server --port 8080

Requirements:
    - LIVEKIT_API_KEY environment variable
    - LIVEKIT_API_SECRET environment variable
    - LIVEKIT_URL environment variable
    - A SIP trunk configured in Livekit
"""

import asyncio
import argparse
import base64
import audioop
import functools
from math import e
import tracemalloc
import wave
import time
import os
import threading
import uvicorn
from typing import Callable, Optional, Any, Dict, List
from datetime import datetime
from uuid import uuid4
from enum import Enum

import numpy as np
from livekit import rtc, api
from livekit.api import AccessToken, VideoGrants
from livekit.protocol.sip import SIPParticipantInfo
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging


# API Models
class CallStatus(str, Enum):
    PENDING = "pending"
    RINGING = "ringing"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CreateCallRequest(BaseModel):
    to_phone: str = Field(..., description="Phone number to call (e.g., +1234567890)")
    wav_file: str = Field(..., description="Path to WAV file to play")
    sip_trunk_id: Optional[str] = Field(None, description="SIP trunk ID (optional)")


class CallResponse(BaseModel):
    call_id: str
    status: CallStatus
    message: str
    to_phone: str
    wav_file: str


class CallStatusResponse(BaseModel):
    call_id: str
    status: CallStatus
    message: str
    participant_connected: bool = False
    audio_playing: bool = False
    progress: Optional[float] = None


# Global call tracking
active_calls: Dict[str, "SimpleLivekitAudioTest"] = {}
call_status_cache: Dict[str, CallStatusResponse] = {}

# FastAPI app
app = FastAPI(
    title="Simple Livekit Audio Test API",
    description="API for creating Livekit calls and playing audio files",
    version="1.0.0",
)


def memory_profiler(top_stats_count: int = 10):
    """
    Decorator to profile memory usage of both sync and async functions.

    Args:
        top_stats_count: Number of top memory statistics to display (default: 10)
        print_stats: Whether to print the statistics (default: True)

    Usage:
        @memory_profiler()
        def sync_function():
            pass

        @memory_profiler(top_stats_count=5)
        async def async_function():
            pass
    """

    def decorator(func: Callable) -> Callable:
        # Check if memory profiling is enabled
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                tracemalloc.start()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    snapshot = tracemalloc.take_snapshot()

                    report_str = print_memory_stats(
                        name=func.__name__,
                        snapshot=snapshot,
                        top_count=top_stats_count,
                    )
                    print(report_str)

                    tracemalloc.stop()

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                tracemalloc.start()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    snapshot = tracemalloc.take_snapshot()

                    report_str = print_memory_stats(
                        name=func.__name__,
                        snapshot=snapshot,
                        top_count=top_stats_count,
                    )
                    print(report_str)

                    tracemalloc.stop()

            return sync_wrapper

    return decorator


def get_memory_snapshot() -> tracemalloc.Snapshot:
    """
    Get current memory snapshot without starting/stopping tracemalloc.
    Useful when tracemalloc is already running.
    """
    return tracemalloc.take_snapshot()


def print_memory_stats(
    name: str, snapshot: tracemalloc.Snapshot, top_count: int = 10
) -> str:
    """
    Print memory statistics from a snapshot.

    Args:
        snapshot: Memory snapshot from tracemalloc
        top_count: Number of top statistics to display
    """
    top_stats = snapshot.statistics("lineno")

    report_str = f"\n--- Memory Statistics ({name}) (Top {top_count}) ---\n"
    for i, stat in enumerate(top_stats[:top_count], 1):
        report_str += f"{i}. {stat}\n"
    report_str += "--- End Memory Statistics ---\n"
    return report_str


class SimpleLivekitAudioTest:
    """Simplified Livekit audio test without complex dependencies"""

    SAMPLE_RATE = 8000
    CHANNELS = 1

    def __init__(
        self, to_phone: str, wav_file: str, sip_trunk_id: Optional[str] = None
    ):
        self.to_phone = to_phone
        self.wav_file = wav_file
        self.sip_trunk_id = sip_trunk_id or os.getenv("LIVEKIT_SIP_TRUNK_ID", "default")

        # Generate unique call ID
        self.call_id = f"test_call_{uuid4().hex[:8]}"
        self.room_name = self.call_id

        # Livekit configuration from environment
        self.livekit_url = os.getenv("LIVEKIT_URL")
        self.livekit_api_key = os.getenv("LIVEKIT_API_KEY")
        self.livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")

        if not all([self.livekit_url, self.livekit_api_key, self.livekit_api_secret]):
            raise ValueError(
                "Missing required environment variables: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET"
            )

        # Initialize Livekit API client
        self.livekit_api = api.LiveKitAPI(
            url=self.livekit_url,
            api_key=self.livekit_api_key,
            api_secret=self.livekit_api_secret,
        )

        # RTC components
        self.rtc_room: Optional[rtc.Room] = None
        self.rtc_source: Optional[rtc.AudioSource] = None
        self.publication: Optional[rtc.LocalTrackPublication] = None
        self.is_call_active = False
        self.participant_connected = False

        # API integration
        self.current_status = CallStatus.PENDING
        self.status_message = "Initialized"
        self.audio_playing = False
        self.audio_progress = 0.0

        print(f"🎯 Initialized test for call {self.call_id}")
        print(f"📞 Calling: {self.to_phone}")
        print(f"🎵 Audio file: {self.wav_file}")

        # Update global status
        self._update_status(CallStatus.PENDING, "Call initialized")

    def _update_status(self, status: CallStatus, message: str):
        """Update call status for API tracking"""
        self.current_status = status
        self.status_message = message

        # Update global cache for API
        call_status_cache[self.call_id] = CallStatusResponse(
            call_id=self.call_id,
            status=status,
            message=message,
            participant_connected=self.participant_connected,
            audio_playing=self.audio_playing,
            progress=self.audio_progress,
        )

    def _create_token(self) -> AccessToken:
        """Create access token for bot participant"""
        token = AccessToken(self.livekit_api_key, self.livekit_api_secret)
        token.identity = f"{self.call_id}_BOT"
        token.with_grants(VideoGrants(room_join=True, room=self.room_name))
        return token

    async def _create_room(self):
        """Create Livekit room"""
        try:
            room = await self.livekit_api.room.create_room(
                api.CreateRoomRequest(name=self.room_name)
            )
            print(f"✅ Created room: {room.name}")
            return room
        except Exception as e:
            print(f"❌ Failed to create room: {e}")
            raise

    async def _create_sip_call(self) -> Optional[SIPParticipantInfo]:
        """Create SIP call participant"""
        try:
            participant = await self.livekit_api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    sip_trunk_id=self.sip_trunk_id,
                    sip_call_to=self.to_phone,
                    room_name=self.room_name,
                    participant_identity=f"{self.call_id}_USER",
                    participant_name=self.to_phone,
                )
            )
            print(f"✅ Created SIP call: {participant.sip_call_id}")
            return participant
        except Exception as e:
            print(f"❌ Failed to create SIP call: {e}")
            raise

    async def _setup_rtc_room(self):
        """Setup RTC room and event handlers"""
        token = self._create_token()

        self.rtc_room = rtc.Room()

        # Event handlers
        self.rtc_room.on("participant_connected", self._on_participant_connected)
        self.rtc_room.on("participant_disconnected", self._on_participant_disconnected)
        self.rtc_room.on(
            "participant_attributes_changed", self._on_participant_attributes_changed
        )
        self.rtc_room.on("disconnected", self._on_room_disconnected)

        # Connect to room
        if not self.livekit_url:
            raise ValueError("LIVEKIT_URL is required")
        await self.rtc_room.connect(self.livekit_url, token.to_jwt())

        if self.rtc_room.isconnected():
            print(f"✅ Connected to RTC room")
        else:
            raise Exception("Failed to connect to RTC room")

    async def _setup_audio_track(self):
        """Setup audio source and track"""
        # Create audio source
        self.rtc_source = rtc.AudioSource(
            loop=asyncio.get_running_loop(),
            sample_rate=self.SAMPLE_RATE,
            num_channels=self.CHANNELS,
        )

        # Create audio track
        track = rtc.LocalAudioTrack.create_audio_track(
            f"{self.call_id}_BOT",
            self.rtc_source,
        )

        # Publish track
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE

        if not self.rtc_room:
            raise RuntimeError("RTC room not initialized")
        self.publication = await self.rtc_room.local_participant.publish_track(
            track, options
        )

        print(f"✅ Published audio track: {self.publication.sid}")

    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        """Handle participant connected"""
        asyncio.create_task(self._on_participant_connected_async(participant))

    async def _on_participant_connected_async(self, participant: rtc.RemoteParticipant):
        print(f"👤 Participant connected: {participant.identity}")

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """Handle participant disconnected"""
        asyncio.create_task(self._on_participant_disconnected_async(participant))

    async def _on_participant_disconnected_async(
        self, participant: rtc.RemoteParticipant
    ):
        print(f"👋 Participant disconnected: {participant.identity}")
        if f"{self.call_id}_USER" in participant.identity:
            print("📞 User hung up, ending call")
            self.is_call_active = False

    def _on_participant_attributes_changed(
        self, changed_attributes: dict, participant: rtc.Participant
    ):
        """Handle participant attribute changes (call status)"""
        asyncio.create_task(
            self._on_participant_attributes_changed_async(
                changed_attributes, participant
            )
        )

    async def _on_participant_attributes_changed_async(
        self, changed_attributes: dict, participant: rtc.Participant
    ):
        sip_call_status = changed_attributes.get("sip.callStatus")

        if sip_call_status == "ringing":
            print("📱 Call is ringing...")
            self._update_status(CallStatus.RINGING, "Call is ringing")
        elif sip_call_status == "active":
            print("✅ Call is active!")
            self.participant_connected = True
            self._update_status(CallStatus.ACTIVE, "Call is active")
        elif sip_call_status == "hangup":
            print("📞 Call ended")
            self.is_call_active = False
            self._update_status(CallStatus.COMPLETED, "Call ended")

    def _on_room_disconnected(self, reason):
        """Handle room disconnection"""
        print(f"🔌 Room disconnected: {reason}")
        self.is_call_active = False

    def _load_wav_file(self) -> np.ndarray:
        """Load WAV file and convert to proper format"""
        if not os.path.exists(self.wav_file):
            self.wav_file = "uploads/sample.wav"

        with wave.open(self.wav_file, "rb") as wf:
            # Check format
            n_channels = wf.getnchannels()
            framerate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()

            print(
                f"🎵 WAV file info: {framerate}Hz, {n_channels}ch, {sampwidth*8}bit, {n_frames} frames"
            )

            # Read audio data
            pcm_data = wf.readframes(n_frames)

            # Convert to mono if needed
            if n_channels == 2:
                pcm_data = audioop.tomono(pcm_data, sampwidth, 0.5, 0.5)
            elif n_channels > 2:
                raise ValueError("WAV file has more than 2 channels")

            # Resample to 8kHz if needed
            if framerate != self.SAMPLE_RATE:
                pcm_data, _ = audioop.ratecv(
                    pcm_data, sampwidth, 1, framerate, self.SAMPLE_RATE, None
                )
                print(f"🔄 Resampled from {framerate}Hz to {self.SAMPLE_RATE}Hz")

            # Convert to numpy array
            if sampwidth == 2:
                audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth}")

            return audio_array

    async def _play_audio_chunks(
        self, audio_array: np.ndarray, end_time: float, chunk_size: int = 160
    ):
        """Play audio in chunks to simulate real-time streaming"""
        print(f"🎵 Playing audio in chunks of {chunk_size} samples...")

        self.audio_playing = True
        self._update_status(CallStatus.ACTIVE, "Playing audio")

        total_chunks = len(audio_array) // chunk_size
        chunk_duration = chunk_size / self.SAMPLE_RATE  # seconds per chunk

        for i in range(0, len(audio_array), chunk_size):
            if not self.is_call_active or not self.participant_connected:
                print("⏹️ Stopping audio playback - call not active")
                self.audio_playing = False
                break

            chunk = audio_array[i : i + chunk_size]

            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(
                    chunk,
                    (0, chunk_size - len(chunk)),
                    mode="constant",
                    constant_values=0,
                )

            # Create audio frame
            frame = rtc.AudioFrame.create(
                sample_rate=self.SAMPLE_RATE,
                num_channels=self.CHANNELS,
                samples_per_channel=len(chunk),
            )

            # Copy audio data to frame
            audio_data = np.frombuffer(frame.data, dtype=np.int16)
            np.copyto(audio_data, chunk)

            # Send to room
            if not self.rtc_source:
                raise RuntimeError("RTC source not initialized")
            await self.rtc_source.capture_frame(frame)

            # Update progress
            progress = (i // chunk_size + 1) / total_chunks * 100
            self.audio_progress = progress / 100.0

            # Progress indicator
            if i % (chunk_size * 50) == 0:  # Every ~1 second
                print(f"🎵 Playing... {self.call_id} {progress:.1f}%")
                self._update_status(
                    CallStatus.ACTIVE, f"Playing audio {self.call_id} ({progress:.1f}%)"
                )

            # Wait for next chunk
            await asyncio.sleep(chunk_duration)
            if time.time() > end_time:
                print(f"🎵 Playing... {self.call_id} {progress:.1f}%")
                self._update_status(
                    CallStatus.ACTIVE, f"Playing audio {self.call_id} ({progress:.1f}%)"
                )
                break

        self.audio_playing = False
        print("✅ Audio playback completed")
        self._update_status(CallStatus.ACTIVE, "Audio playback completed")

    async def _stop_recording(self):
        """Stop recording if active"""
        # Note: This is a placeholder for recording functionality
        # In the full codebase, this would stop egress recording
        print("📹 Stopping recording (if active)")

    async def _end_call(self):
        """End the SIP call"""
        if not self.room_name:
            return

        try:
            # End the call via SIP API - this is similar to livekit_connector._end_call()
            print(f"📞 Ending SIP call for room {self.room_name}")
            # Note: The actual SIP call termination is handled by room deletion
            # but in production you might want more granular control
        except Exception as e:
            print(f"⚠️ Failed to end SIP call: {e}")

    async def _disconnect_from_rtc_room(self):
        """Disconnect from RTC room - similar to livekit_connector._disconnect_from_rtc_room()"""
        if not self.rtc_room:
            print("🔌 RTC room is already disconnected")
            return

        try:
            print("🔌 Disconnecting from RTC room...")
            await asyncio.wait_for(
                self.rtc_room.disconnect(),
                timeout=30,
            )
            print("✅ Disconnected from RTC room")
        except Exception as e:
            print(f"⚠️ Failed to disconnect from RTC room: {e}")

    async def _cancel_audio_track_tasks(self):
        """Cancel audio tracks and cleanup - similar to livekit_connector._cancel_audio_track_tasks()"""
        if self.rtc_source:
            try:
                print("🎵 Clearing audio source queue...")
                self.rtc_source.clear_queue()

                # Close the audio source
                await asyncio.wait_for(
                    self.rtc_source.aclose(),
                    timeout=10,
                )
                print("✅ Audio source cleaned up")
            except Exception as e:
                print(f"⚠️ Failed to cleanup audio source: {e}")

    async def _delete_room(self):
        """Delete the room via API"""
        try:
            await self.livekit_api.room.delete_room(
                api.DeleteRoomRequest(room=self.room_name)
            )
            print("✅ Room deleted")
        except Exception as e:
            print(f"⚠️ Failed to delete room: {e}")

    async def _cleanup(self):
        """
        Comprehensive cleanup - following the pattern from livekit_connector.end_room()
        and livekit_call_stream_relay_service._clean_up_all_resources()
        """
        print("🧹 Starting comprehensive cleanup...")

        # 1. Stop recording (if implemented)
        await self._stop_recording()

        # 2. Set call as ended to stop any ongoing processes
        self.is_call_active = False
        self.audio_playing = False

        # 3. Disconnect from RTC room
        await self._disconnect_from_rtc_room()

        # 4. End the SIP call
        await self._end_call()

        # 5. Cancel and cleanup audio tracks
        await self._cancel_audio_track_tasks()

        # 6. Delete the room
        await self._delete_room()

        print("✅ Comprehensive cleanup completed")

    async def cancel_call(self):
        """
        Cancel an active call - similar to stopping mechanisms in the original codebase
        This can be called from the API or internally
        """
        print("🛑 Cancelling call...")

        # Stop all active processes
        self.is_call_active = False
        self.audio_playing = False
        self.participant_connected = False

        # Update status
        self._update_status(CallStatus.CANCELLED, "Call cancelled")

        # Cleanup all resources
        await self._cleanup()

    @memory_profiler()
    async def run(self):
        """Main execution flow"""
        try:
            print("🚀 Starting Livekit audio test...")

            # 1. Create room
            await self._create_room()

            # 2. Setup RTC connection
            await self._setup_rtc_room()

            # 3. Setup audio track
            await self._setup_audio_track()

            # 4. Create SIP call
            await self._create_sip_call()

            # 5. Load audio file
            audio_data = await asyncio.to_thread(self._load_wav_file)
            print(
                f"✅ Loaded audio: {len(audio_data)} samples, {len(audio_data)/self.SAMPLE_RATE:.1f}s duration"
            )

            # 6. Wait for participant to connect
            self.is_call_active = True
            print("⏳ Waiting for call to be answered...")

            # Wait up to 30 seconds for call to be answered
            timeout = 30

            start_time = time.time()
            end_time = start_time + timeout
            while (
                not self.participant_connected
                and self.is_call_active
                and (time.time() - start_time) < timeout
            ):
                print(f"Waiting for call to be answered... {self.call_id}")
                await asyncio.sleep(0.5)

            if not self.participant_connected:
                print("❌ Call was not answered within timeout")
                return

            # 7. Play audio
            await self._play_audio_chunks(audio_data, end_time=end_time)

            # 8. Keep call alive for a bit after playback
            print("⏳ Keeping call alive for 3 more seconds...")
            await asyncio.sleep(3)

        except Exception as e:
            logging.error(f"Error during execution: {e}")
            self._update_status(CallStatus.FAILED, f"Error: {str(e)}")
            raise
        finally:
            await self._cleanup()
            print("✅ Test completed")
            if self.current_status != CallStatus.FAILED:
                self._update_status(CallStatus.COMPLETED, "Test completed successfully")


# FastAPI Endpoints


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Simple Livekit Audio Test API", "status": "running"}


@app.get("/calls", response_model=List[CallStatusResponse])
async def list_calls():
    """List all active and recent calls"""
    return list(call_status_cache.values())


@app.post("/calls", response_model=CallResponse)
async def create_call(request: CreateCallRequest, background_tasks: BackgroundTasks):
    """Create a new Livekit call with audio playback"""

    # Validate WAV file exists
    if not os.path.exists(request.wav_file):
        raise HTTPException(
            status_code=400, detail=f"WAV file not found: {request.wav_file}"
        )

    # Validate environment variables
    required_env_vars = ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise HTTPException(
            status_code=500,
            detail=f"Missing required environment variables: {', '.join(missing_vars)}",
        )

    try:
        # Create test instance
        test = SimpleLivekitAudioTest(
            to_phone=request.to_phone,
            wav_file=request.wav_file,
            sip_trunk_id=request.sip_trunk_id,
        )

        # Store in active calls
        active_calls[test.call_id] = test

        # Run test in background
        background_tasks.add_task(run_call_background, test)

        return CallResponse(
            call_id=test.call_id,
            status=CallStatus.PENDING,
            message="Call initiated",
            to_phone=request.to_phone,
            wav_file=request.wav_file,
        )

    except Exception as e:
        logging.error(f"Failed to create call: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create call: {str(e)}")


@app.get("/calls/{call_id}", response_model=CallStatusResponse)
async def get_call_status(call_id: str):
    """Get status of a specific call"""
    if call_id not in call_status_cache:
        raise HTTPException(status_code=404, detail="Call not found")

    return call_status_cache[call_id]


@app.delete("/calls/{call_id}")
async def cancel_call_api(call_id: str, background_tasks: BackgroundTasks):
    """Cancel an active call"""
    if call_id not in active_calls:
        raise HTTPException(
            status_code=404, detail="Call not found or already completed"
        )

    test = active_calls[call_id]

    # Cancel the call in background to avoid blocking the API response
    background_tasks.add_task(test.cancel_call)

    return {"message": f"Call {call_id} cancellation initiated"}


@app.post("/upload-audio")
async def upload_audio_file(file: UploadFile = File(...)):
    """Upload a WAV file for use in calls"""
    if not file.filename or not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    # Create uploads directory if it doesn't exist
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Save uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "path": file_path,
        "size": len(content),
    }


async def run_call_background(test: SimpleLivekitAudioTest):
    """Run a call in the background"""
    try:
        await test.run()
    except Exception as e:
        print(f"Background call failed: {e}")
        test._update_status(CallStatus.FAILED, f"Background call failed: {str(e)}")
    finally:
        # Remove from active calls when done
        if test.call_id in active_calls:
            del active_calls[test.call_id]


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server"""
    print(f"🚀 Starting Livekit Audio Test API server on {host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/")
    print(f"App PID: {os.getpid()}")

    uvicorn.run(app, host=host, port=port)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Simple Livekit audio test with API server"
    )

    # Server mode arguments
    parser.add_argument("--server", action="store_true", help="Run as FastAPI server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )

    # CLI mode arguments
    parser.add_argument("--to-phone", help="Phone number to call (e.g., +1234567890)")
    parser.add_argument("--wav-file", help="Path to WAV file to play")
    parser.add_argument(
        "--sip-trunk-id", help="SIP trunk ID (defaults to env LIVEKIT_SIP_TRUNK_ID)"
    )

    args = parser.parse_args()

    print("Parent PID: ", os.getppid())

    # Server mode
    if args.server:
        run_server(host=args.host, port=args.port)
        return 0

    # CLI mode - require phone and wav file
    if not args.to_phone or not args.wav_file:
        parser.error(
            "--to-phone and --wav-file are required for CLI mode. Use --server for API mode."
        )

    # Validate WAV file exists
    if not os.path.exists(args.wav_file):
        print(f"❌ WAV file not found: {args.wav_file}")
        return 1

    # Create and run test
    test = SimpleLivekitAudioTest(
        to_phone=args.to_phone, wav_file=args.wav_file, sip_trunk_id=args.sip_trunk_id
    )

    try:
        asyncio.run(test.run())
        return 0
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
