"""
Real-time monitor for EBM rerank activity
Run this in a separate terminal to see when rerank is being called
"""

import time
import subprocess
import sys
from datetime import datetime
from collections import deque
import requests

# Configuration
FASTAPI_URL = "http://127.0.0.1:8000"
CHECK_INTERVAL = 2  # seconds

# Track request count
request_count = 0
last_check = datetime.now()


def check_api_health():
    """Check if FastAPI is running"""
    try:
        resp = requests.get(f"{FASTAPI_URL}/health", timeout=2)
        return resp.status_code == 200
    except:
        return False


def print_header():
    """Print monitoring header"""
    print("\n" + "=" * 80)
    print("EBM RERANK ACTIVITY MONITOR")
    print("=" * 80)
    print(f"FastAPI URL: {FASTAPI_URL}")
    print(f"Started: {datetime.now()}")
    print("=" * 80)
    print("\nWaiting for rerank requests...\n")


def print_status(api_status, count):
    """Print current status"""
    status_icon = "🟢" if api_status else "🔴"
    print(f"\r{status_icon} API Status: {'ONLINE' if api_status else 'OFFLINE'} | "
          f"Requests detected: {count} | "
          f"Time: {datetime.now().strftime('%H:%M:%S')}", end="", flush=True)


def monitor_logs():
    """Monitor RAGFlow logs for EBM rerank activity"""
    global request_count

    print_header()

    # Try to find RAGFlow log file
    log_paths = [
        "/media/bazzi/Bazzi/GitHubBazzi/ragflow/logs/api.log",
        "logs/api.log",
    ]

    log_file = None
    for path in log_paths:
        try:
            with open(path, 'r') as f:
                log_file = path
                break
        except FileNotFoundError:
            continue

    if not log_file:
        print("\nRAGFlow log file not found. Monitoring API health only.\n")

    # Monitor loop
    log_buffer = deque(maxlen=20)  # Keep last 20 log lines

    if log_file:
        # Start tailing the log file
        try:
            with subprocess.Popen(
                ['tail', '-f', log_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            ) as proc:

                while True:
                    # Check API health
                    api_status = check_api_health()

                    # Read log lines
                    line = proc.stdout.readline()
                    if line:
                        # Check for EBM rerank activity
                        if "[EBM RERANK]" in line:
                            request_count += 1
                            print("\n" + "-" * 80)
                            print(f"RERANK DETECTED! (#{request_count})")
                            print(f"{datetime.now()}")
                            print("-" * 80)
                            print(line.strip())

                            # Print context (previous and next few lines)
                            for buffered_line in log_buffer:
                                if "EBM" in buffered_line or "rerank" in buffered_line.lower():
                                    print(buffered_line.strip())

                            print("-" * 80 + "\n")

                        log_buffer.append(line)

                    print_status(api_status, request_count)
                    time.sleep(0.1)

        except FileNotFoundError:
            print(f"\nCannot tail log file: {log_file}")
            print("   Make sure RAGFlow is running and generating logs.\n")
    else:
        # Just monitor API health
        while True:
            api_status = check_api_health()
            print_status(api_status, request_count)
            time.sleep(CHECK_INTERVAL)


def main():
    """Main entry point"""
    try:
        print("\nStarting EBM Rerank Monitor...")
        print("   Press Ctrl+C to stop\n")
        time.sleep(1)
        monitor_logs()
    except KeyboardInterrupt:
        print("\n\nMonitor stopped by user")
        print(f"Total rerank requests detected: {request_count}")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()