# LiveKit Load Test Tools

A comprehensive toolkit for load testing and benchmarking LiveKit audio calls. This project provides both a FastAPI server for creating LiveKit calls with audio playback and advanced benchmarking tools to measure performance under various load conditions.

## Features

### 🎯 Core Functionality

- **LiveKit Audio Call Server**: FastAPI-based server for creating and managing LiveKit calls
- **SIP Integration**: Support for SIP trunk calls with configurable parameters
- **Audio Playback**: Play WAV files during calls with real-time audio streaming
- **Call Management**: Complete call lifecycle management (create, monitor, cancel)
- **File Upload**: Upload and manage audio files via REST API

### 📊 Load Testing & Benchmarking

- **Sustained Load Testing**: Maintain target calls per second (CPS) over time
- **Burst Testing**: Generate high-volume call bursts to test peak capacity
- **Ramp-up Testing**: Gradually increase load to find breaking points
- **Performance Metrics**: Comprehensive latency, throughput, and system metrics
- **Real-time Monitoring**: System resource monitoring during tests

### 🛠 Management Tools

- **Room Cleanup**: Automated cleanup of LiveKit rooms
- **Memory Profiling**: Built-in memory usage tracking and profiling
- **Background Tasks**: Asynchronous call processing with status tracking

## Requirements

- Python 3.11+
- Poetry for dependency management
- LiveKit server with SIP trunk configuration
- Audio files in WAV format (8kHz, mono recommended)

### Environment Variables

```bash
# Required LiveKit configuration
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret

# Optional SIP configuration
LIVEKIT_SIP_TRUNK_ID=your-sip-trunk-id
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/thinhdanggroup/livekit-loadtest-tools.git
   cd livekit-loadtest-tools
   ```

2. **Install dependencies**:

   ```bash
   make install
   # or
   poetry install --no-root
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your LiveKit configuration
   ```

## Quick Start

### 1. Start the API Server

```bash
# Using Makefile
make benchmark-server-up

# Or directly
poetry run python simple_livekit_audio_test.py --server --port 8081
```

The server will start on `http://localhost:8081` with:

- **API Documentation**: http://localhost:8081/docs
- **Health Check**: http://localhost:8081/

### 2. Run Load Tests

```bash
# Sustained load test (1 CPS for 10 seconds)
make benchmark-start

# Custom sustained test
poetry run python benchmark_livekit_calls.py \
  --sustained \
  --target-cps 5 \
  --duration 60 \
  --phone +1234567890 \
  --api-url http://localhost:8081 \
  --wav-file uploads/sample.wav

# Burst test (100 calls as fast as possible)
poetry run python benchmark_livekit_calls.py \
  --burst \
  --burst-size 100 \
  --max-concurrent 20 \
  --api-url http://localhost:8081

# Ramp-up test (gradually increase to 10 CPS)
poetry run python benchmark_livekit_calls.py \
  --ramp-up \
  --max-cps 10 \
  --ramp-duration 30 \
  --sustain-duration 60 \
  --api-url http://localhost:8081
```

### 3. Clean Up Resources

```bash
# Clean up LiveKit rooms
make benchmark-clean-up

# Or directly
poetry run pytest clean_up_room.py
```

## API Reference

### Create a Call

```bash
POST /calls
Content-Type: application/json

{
  "to_phone": "+1234567890",
  "wav_file": "uploads/sample.wav",
  "sip_trunk_id": "optional-trunk-id"
}
```

**Response**:

```json
{
  "call_id": "test_call_a1b2c3d4",
  "status": "pending",
  "message": "Call initiated",
  "to_phone": "+1234567890",
  "wav_file": "uploads/sample.wav"
}
```

### Get Call Status

```bash
GET /calls/{call_id}
```

**Response**:

```json
{
  "call_id": "test_call_a1b2c3d4",
  "status": "active",
  "message": "Call in progress",
  "participant_connected": true,
  "audio_playing": true,
  "progress": 0.45
}
```

### List All Calls

```bash
GET /calls
```

### Cancel a Call

```bash
DELETE /calls/{call_id}
```

### Upload Audio File

```bash
POST /upload-audio
Content-Type: multipart/form-data

file: <wav-file>
```

## Usage Examples

### CLI Mode (Direct Call)

```bash
# Make a single call with CLI
poetry run python simple_livekit_audio_test.py \
  --to-phone "+1234567890" \
  --wav-file "uploads/payment_init.wav" \
  --sip-trunk-id "my-trunk"
```

### API Mode (Server)

```bash
# Start server on custom host/port
poetry run python simple_livekit_audio_test.py \
  --server \
  --host 0.0.0.0 \
  --port 8080
```

### Advanced Benchmarking

```bash
# Comprehensive load test with output
poetry run python benchmark_livekit_calls.py \
  --sustained \
  --target-cps 15 \
  --duration 300 \
  --max-concurrent 50 \
  --phone "+1234567890" \
  --wav-file "uploads/sample.wav" \
  --api-url http://localhost:8081 \
  --output results.json

# Test with multiple concurrent connections
poetry run python benchmark_livekit_calls.py \
  --burst \
  --burst-size 500 \
  --max-concurrent 100 \
  --timeout 60 \
  --output burst_test.json
```

## Performance Metrics

The benchmark tool provides comprehensive metrics:

### Call Metrics

- **Total Calls**: Number of call attempts
- **Success Rate**: Percentage of successful calls
- **Calls Per Second**: Actual throughput achieved
- **Latency Statistics**: Average, min, max, 95th, 99th percentiles

### System Metrics

- **CPU Usage**: Processor utilization during tests
- **Memory Usage**: RAM consumption and availability
- **Network I/O**: Data sent/received during tests
- **Disk Usage**: Storage utilization

### Example Output

```
============================================================
📊 BENCHMARK RESULTS
============================================================
🎯 Total Calls: 1
✅ Successful: 1
❌ Failed: 0
📈 Success Rate: 100.00%
⏱️  Duration: 10.01s
🚀 Calls/Second: 0.10

📡 LATENCY METRICS
   Average: 23.69ms
   Minimum: 23.69ms
   Maximum: 23.69ms
   95th percentile: 23.69ms
   99th percentile: 23.69ms

💻 SYSTEM METRICS
   avg_cpu_percent: 39.1%
   max_cpu_percent: 50.9%
   avg_memory_percent: 80.3%
   max_memory_percent: 80.5%
   avg_memory_used_gb: 5.80GB
   max_memory_used_gb: 5.85GB
   avg_memory_available_gb: 3.16GB
   max_memory_available_gb: 3.21GB
   avg_disk_percent: 5.7%
   max_disk_percent: 5.7%
   avg_network_sent_mb: 7639.7MB
   max_network_sent_mb: 7641.3MB
   avg_network_recv_mb: 7267.9MB
   max_network_recv_mb: 7270.3MB
```

## File Structure

```
livekit-loadtest-tools/
├── simple_livekit_audio_test.py    # Main FastAPI server
├── benchmark_livekit_calls.py      # Load testing tool
├── clean_up_room.py                # Room cleanup utility
├── pyproject.toml                  # Project dependencies
├── Makefile                        # Convenient commands
├── uploads/                        # Audio files directory
│   ├── sample.wav                  # Sample audio file
└── README.md                       # This file
```

## Audio File Requirements

- **Format**: WAV files only
- **Sample Rate**: 8kHz (recommended for telephony)
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit recommended

The system automatically handles:

- Audio format conversion for LiveKit compatibility
- Real-time audio streaming with proper timing
- Audio chunk processing (160 samples per chunk at 8kHz)

## Troubleshooting

### Common Issues

1. **Environment Variables Not Set**:

   ```
   Missing required environment variables: LIVEKIT_URL, LIVEKIT_API_KEY
   ```

   Solution: Set all required environment variables

2. **Audio File Not Found**:

   ```
   WAV file not found: uploads/sample.wav
   ```

   Solution: Upload audio file via API or place in uploads/ directory

3. **Connection Timeout**:

   ```
   Failed to connect to API: Cannot connect to host localhost:8081
   ```

   Solution: Ensure the API server is running before starting benchmarks

4. **SIP Trunk Issues**:
   - Verify SIP trunk configuration in LiveKit
   - Check phone number format (must include country code)
   - Ensure SIP trunk ID is correct

### Memory Issues

If experiencing memory issues during load tests:

- Reduce concurrent connections (`--max-concurrent`)
- Monitor system resources during tests
- Consider running tests on a machine with more RAM

## Contributing

1. Follow the existing code structure for unit tests
2. Ensure all tests pass before submitting PRs
3. Add appropriate logging and error handling
4. Update documentation for new features

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

**thinhdanggroup** (dinhluchvc@gmail.com)

---

For more detailed API documentation, visit http://localhost:8081/docs when the server is running.
