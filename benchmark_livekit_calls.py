#!/usr/bin/env python3
"""
Livekit Calls Per Second (CPS) Benchmark Script

This script benchmarks the performance of creating Livekit calls through the API
to measure calls per second, latency, and system performance.

Usage:
    python benchmark_livekit_calls.py --target-cps 10 --duration 60
    python benchmark_livekit_calls.py --concurrent-calls 50 --ramp-up 30
    python benchmark_livekit_calls.py --burst-test --burst-size 100

Requirements:
    - The Livekit API server must be running
    - Test audio files available
    - Valid phone numbers for testing (or use fake numbers for load testing)
"""

import asyncio
import aiohttp
import argparse
import time
import json
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import psutil
import sys
import signal
from pathlib import Path


@dataclass
class CallResult:
    """Result of a single call attempt"""
    success: bool
    call_id: Optional[str]
    latency_ms: float
    status_code: int
    error_message: Optional[str]
    timestamp: datetime


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark metrics"""
    total_calls: int
    successful_calls: int
    failed_calls: int
    success_rate: float
    duration_seconds: float
    calls_per_second: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    errors: Dict[str, int]
    system_metrics: Dict[str, float]


class LivekitBenchmark:
    """Livekit calls per second benchmark"""
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        default_phone: str = "+1234567890",  # Use fake number for load testing
        default_wav_file: str = "test_audio.wav",
        timeout: int = 30
    ):
        self.api_base_url = api_base_url.rstrip('/')
        self.default_phone = default_phone
        self.default_wav_file = default_wav_file
        self.timeout = timeout
        
        self.results: List[CallResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.running = False
        
        # System monitoring
        self.system_metrics_history: List[Dict[str, float]] = []
        
        print(f"🎯 Livekit CPS Benchmark initialized")
        print(f"📡 API URL: {self.api_base_url}")
        print(f"📞 Default phone: {self.default_phone}")
        print(f"🎵 Default audio: {self.default_wav_file}")

    async def _make_call_request(
        self, 
        session: aiohttp.ClientSession, 
        phone: Optional[str] = None, 
        wav_file: Optional[str] = None
    ) -> CallResult:
        """Make a single call request and measure performance"""
        phone = phone or self.default_phone
        wav_file = wav_file or self.default_wav_file
        
        request_data = {
            "to_phone": phone,
            "wav_file": wav_file
        }
        
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            async with session.post(
                f"{self.api_base_url}/calls",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                latency_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    response_data = await response.json()
                    return CallResult(
                        success=True,
                        call_id=response_data.get("call_id"),
                        latency_ms=latency_ms,
                        status_code=response.status,
                        error_message=None,
                        timestamp=timestamp
                    )
                else:
                    error_text = await response.text()
                    return CallResult(
                        success=False,
                        call_id=None,
                        latency_ms=latency_ms,
                        status_code=response.status,
                        error_message=f"HTTP {response.status}: {error_text}",
                        timestamp=timestamp
                    )
                    
        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            return CallResult(
                success=False,
                call_id=None,
                latency_ms=latency_ms,
                status_code=0,
                error_message="Request timeout",
                timestamp=timestamp
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return CallResult(
                success=False,
                call_id=None,
                latency_ms=latency_ms,
                status_code=0,
                error_message=str(e),
                timestamp=timestamp
            )

    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system performance metrics"""
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network stats (if available)
        try:
            net_io = psutil.net_io_counters()
            network_sent_mb = net_io.bytes_sent / (1024 * 1024)
            network_recv_mb = net_io.bytes_recv / (1024 * 1024)
        except:
            network_sent_mb = 0
            network_recv_mb = 0
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "network_sent_mb": network_sent_mb,
            "network_recv_mb": network_recv_mb,
        }

    async def _system_monitor_task(self):
        """Background task to monitor system metrics"""
        while self.running:
            metrics = self._collect_system_metrics()
            metrics["timestamp"] = time.time()
            self.system_metrics_history.append(metrics)
            await asyncio.sleep(1)  # Collect metrics every second

    async def sustained_load_test(
        self, 
        target_cps: float, 
        duration_seconds: int,
        max_concurrent: int = 100
    ) -> BenchmarkMetrics:
        """Run sustained load test with target CPS"""
        print(f"🚀 Starting sustained load test:")
        print(f"   📊 Target CPS: {target_cps}")
        print(f"   ⏱️  Duration: {duration_seconds}s")
        print(f"   🔄 Max concurrent: {max_concurrent}")
        
        self.running = True
        self.start_time = datetime.now()
        self.results = []
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self._system_monitor_task())
        
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Calculate interval between requests
            interval = 1.0 / target_cps if target_cps > 0 else 0
            
            end_time = time.time() + duration_seconds
            tasks = []
            
            while time.time() < end_time and self.running:
                # Launch new request
                task = asyncio.create_task(self._make_call_request(session))
                tasks.append(task)
                
                # Clean up completed tasks
                tasks = [t for t in tasks if not t.done()]
                
                # Collect completed results
                for task in [t for t in tasks if t.done()]:
                    try:
                        result = await task
                        self.results.append(result)
                    except Exception as e:
                        print(f"⚠️ Task error: {e}")
                
                # Limit concurrent requests
                if len(tasks) >= max_concurrent:
                    await asyncio.sleep(0.1)
                    continue
                
                # Wait for next interval
                if interval > 0:
                    await asyncio.sleep(interval)
            
            # Wait for remaining tasks
            print("⏳ Waiting for remaining requests to complete...")
            if tasks:
                remaining_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in remaining_results:
                    if isinstance(result, CallResult):
                        self.results.append(result)
        
        self.running = False
        self.end_time = datetime.now()
        
        # Stop monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        return self._calculate_metrics()

    async def burst_test(self, burst_size: int, concurrent_limit: Optional[int] = None) -> BenchmarkMetrics:
        """Run burst test - create many calls as fast as possible"""
        concurrent_limit = concurrent_limit or min(burst_size, 100)
        
        print(f"💥 Starting burst test:")
        print(f"   📊 Burst size: {burst_size}")
        print(f"   🔄 Concurrent limit: {concurrent_limit}")
        
        self.running = True
        self.start_time = datetime.now()
        self.results = []
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self._system_monitor_task())
        
        connector = aiohttp.TCPConnector(limit=concurrent_limit)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create all tasks
            tasks = [
                asyncio.create_task(self._make_call_request(session)) 
                for _ in range(burst_size)
            ]
            
            # Execute with concurrency limit
            semaphore = asyncio.Semaphore(concurrent_limit)
            
            async def limited_call():
                async with semaphore:
                    return await self._make_call_request(session)
            
            tasks = [asyncio.create_task(limited_call()) for _ in range(burst_size)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, CallResult):
                    self.results.append(result)
                else:
                    print(f"⚠️ Burst task error: {result}")
        
        self.running = False
        self.end_time = datetime.now()
        
        # Stop monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        return self._calculate_metrics()

    async def ramp_up_test(
        self, 
        max_cps: float, 
        ramp_duration: int, 
        sustain_duration: int = 60
    ) -> BenchmarkMetrics:
        """Run ramp-up test - gradually increase CPS to max"""
        print(f"📈 Starting ramp-up test:")
        print(f"   📊 Max CPS: {max_cps}")
        print(f"   ⬆️ Ramp duration: {ramp_duration}s")
        print(f"   ⏱️ Sustain duration: {sustain_duration}s")
        
        self.running = True
        self.start_time = datetime.now()
        self.results = []
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self._system_monitor_task())
        
        connector = aiohttp.TCPConnector(limit=200)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Ramp-up phase
            ramp_start = time.time()
            last_request_time = ramp_start
            
            while time.time() - ramp_start < ramp_duration and self.running:
                elapsed = time.time() - ramp_start
                progress = elapsed / ramp_duration
                current_cps = max_cps * progress
                
                if current_cps > 0:
                    interval = 1.0 / current_cps
                    if time.time() - last_request_time >= interval:
                        task = asyncio.create_task(self._make_call_request(session))
                        # Don't await - let it run in background
                        asyncio.create_task(self._collect_result(task))
                        last_request_time = time.time()
                
                await asyncio.sleep(0.01)  # Small sleep to prevent tight loop
            
            # Sustain phase
            print(f"📊 Ramped up to {max_cps} CPS, sustaining for {sustain_duration}s...")
            sustain_start = time.time()
            interval = 1.0 / max_cps if max_cps > 0 else 0
            
            while time.time() - sustain_start < sustain_duration and self.running:
                task = asyncio.create_task(self._make_call_request(session))
                asyncio.create_task(self._collect_result(task))
                
                if interval > 0:
                    await asyncio.sleep(interval)
        
        self.running = False
        self.end_time = datetime.now()
        
        # Stop monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        # Wait a bit for any remaining requests
        await asyncio.sleep(2)
        
        return self._calculate_metrics()

    async def _collect_result(self, task: asyncio.Task):
        """Collect result from a background task"""
        try:
            result = await task
            self.results.append(result)
        except Exception as e:
            print(f"⚠️ Background task error: {e}")

    def _calculate_metrics(self) -> BenchmarkMetrics:
        """Calculate benchmark metrics from results"""
        if not self.results:
            return BenchmarkMetrics(
                total_calls=0, successful_calls=0, failed_calls=0,
                success_rate=0.0, duration_seconds=0.0, calls_per_second=0.0,
                avg_latency_ms=0.0, min_latency_ms=0.0, max_latency_ms=0.0,
                p95_latency_ms=0.0, p99_latency_ms=0.0,
                errors={}, system_metrics={}
            )
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        # Calculate duration
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        else:
            duration = 0
        
        # Calculate latency statistics
        latencies = [r.latency_ms for r in self.results]
        latencies.sort()
        
        # Calculate percentiles
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = k - f
            if f == len(data) - 1:
                return data[f]
            return data[f] * (1 - c) + data[f + 1] * c
        
        # Error analysis
        errors = {}
        for result in failed:
            error_key = result.error_message or f"HTTP {result.status_code}"
            errors[error_key] = errors.get(error_key, 0) + 1
        
        # System metrics
        avg_system_metrics = {}
        if self.system_metrics_history:
            metric_keys = self.system_metrics_history[0].keys()
            for key in metric_keys:
                if key != "timestamp":
                    values = [m[key] for m in self.system_metrics_history]
                    avg_system_metrics[f"avg_{key}"] = statistics.mean(values)
                    avg_system_metrics[f"max_{key}"] = max(values)
        
        return BenchmarkMetrics(
            total_calls=len(self.results),
            successful_calls=len(successful),
            failed_calls=len(failed),
            success_rate=(len(successful) / len(self.results)) * 100 if self.results else 0,
            duration_seconds=duration,
            calls_per_second=len(self.results) / duration if duration > 0 else 0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            p95_latency_ms=percentile(latencies, 95),
            p99_latency_ms=percentile(latencies, 99),
            errors=errors,
            system_metrics=avg_system_metrics
        )

    def print_metrics(self, metrics: BenchmarkMetrics):
        """Print formatted benchmark results"""
        print("\n" + "="*60)
        print("📊 BENCHMARK RESULTS")
        print("="*60)
        
        print(f"🎯 Total Calls: {metrics.total_calls}")
        print(f"✅ Successful: {metrics.successful_calls}")
        print(f"❌ Failed: {metrics.failed_calls}")
        print(f"📈 Success Rate: {metrics.success_rate:.2f}%")
        print(f"⏱️  Duration: {metrics.duration_seconds:.2f}s")
        print(f"🚀 Calls/Second: {metrics.calls_per_second:.2f}")
        
        print(f"\n📡 LATENCY METRICS")
        print(f"   Average: {metrics.avg_latency_ms:.2f}ms")
        print(f"   Minimum: {metrics.min_latency_ms:.2f}ms")
        print(f"   Maximum: {metrics.max_latency_ms:.2f}ms")
        print(f"   95th percentile: {metrics.p95_latency_ms:.2f}ms")
        print(f"   99th percentile: {metrics.p99_latency_ms:.2f}ms")
        
        if metrics.errors:
            print(f"\n❌ ERROR BREAKDOWN")
            for error, count in metrics.errors.items():
                print(f"   {error}: {count}")
        
        if metrics.system_metrics:
            print(f"\n💻 SYSTEM METRICS")
            for metric, value in metrics.system_metrics.items():
                if "percent" in metric:
                    print(f"   {metric}: {value:.1f}%")
                elif "gb" in metric:
                    print(f"   {metric}: {value:.2f}GB")
                elif "mb" in metric:
                    print(f"   {metric}: {value:.1f}MB")
                else:
                    print(f"   {metric}: {value:.2f}")

    def save_results(self, filename: str, metrics: BenchmarkMetrics):
        """Save detailed results to JSON file"""
        output_data = {
            "metrics": asdict(metrics),
            "detailed_results": [asdict(r) for r in self.results],
            "system_history": self.system_metrics_history,
            "benchmark_info": {
                "api_base_url": self.api_base_url,
                "default_phone": self.default_phone,
                "default_wav_file": self.default_wav_file,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"💾 Results saved to {filename}")

    def stop(self):
        """Stop the benchmark"""
        self.running = False
        print("\n🛑 Benchmark stopped by user")


# Global benchmark instance for signal handling
benchmark_instance = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    if benchmark_instance:
        benchmark_instance.stop()

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


async def main():
    """Main benchmark function"""
    global benchmark_instance
    
    parser = argparse.ArgumentParser(description="Livekit CPS Benchmark Tool")
    
    # Connection settings
    parser.add_argument("--api-url", default="http://localhost:8000", 
                       help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--phone", default="+1234567890", 
                       help="Phone number for testing (default: +1234567890)")
    parser.add_argument("--wav-file", default="test_audio.wav", 
                       help="WAV file path (default: test_audio.wav)")
    parser.add_argument("--timeout", type=int, default=30, 
                       help="Request timeout in seconds (default: 30)")
    
    # Test types
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument("--sustained", action="store_true", 
                           help="Run sustained load test")
    test_group.add_argument("--burst", action="store_true", 
                           help="Run burst test")
    test_group.add_argument("--ramp-up", action="store_true", 
                           help="Run ramp-up test")
    
    # Sustained test options
    parser.add_argument("--target-cps", type=float, default=1, 
                       help="Target calls per second (default: 1)")
    parser.add_argument("--duration", type=int, default=60, 
                       help="Test duration in seconds (default: 60)")
    parser.add_argument("--max-concurrent", type=int, default=10, 
                       help="Max concurrent requests (default: 10)")
    
    # Burst test options
    parser.add_argument("--burst-size", type=int, default=100, 
                       help="Number of calls in burst (default: 100)")
    
    # Ramp-up test options
    parser.add_argument("--max-cps", type=float, default=50, 
                       help="Maximum CPS for ramp-up (default: 50)")
    parser.add_argument("--ramp-duration", type=int, default=60, 
                       help="Ramp-up duration in seconds (default: 60)")
    parser.add_argument("--sustain-duration", type=int, default=60, 
                       help="Sustain duration in seconds (default: 60)")
    
    # Output options
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark_instance = LivekitBenchmark(
        api_base_url=args.api_url,
        default_phone=args.phone,
        default_wav_file=args.wav_file,
        timeout=args.timeout
    )
    
    # Check if API is accessible
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.api_url}/") as response:
                if response.status != 200:
                    print(f"❌ API not accessible at {args.api_url}")
                    return 1
                print(f"✅ API accessible at {args.api_url}")
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        return 1
    
    # Check if WAV file exists
    if not Path(args.wav_file).exists():
        print(f"⚠️ WAV file not found: {args.wav_file}")
        print("   Continuing anyway - the API will handle the error")
    
    # Run appropriate test
    try:
        metrics = None
        if args.sustained:
            metrics = await benchmark_instance.sustained_load_test(
                target_cps=args.target_cps,
                duration_seconds=args.duration,
                max_concurrent=args.max_concurrent
            )
        elif args.burst:
            metrics = await benchmark_instance.burst_test(
                burst_size=args.burst_size,
                concurrent_limit=args.max_concurrent
            )
        elif args.ramp_up:
            metrics = await benchmark_instance.ramp_up_test(
                max_cps=args.max_cps,
                ramp_duration=args.ramp_duration,
                sustain_duration=args.sustain_duration
            )
        
        # Print results
        if metrics:
            benchmark_instance.print_metrics(metrics)
            
            # Save results if requested
            if args.output:
                benchmark_instance.save_results(args.output, metrics)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main())) 