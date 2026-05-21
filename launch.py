#!/usr/bin/env python3
"""
Sodil — One-command launcher
────────────────────────────
Just run:  python launch.py

What it does:
  1. Installs all dependencies automatically
  2. Starts the web dashboard
  3. Opens your browser to http://localhost:8501
"""
import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent
PORT = 8501
URL = f"http://localhost:{PORT}"


def main():
    print("\n📈 Sodil — Investment Intelligence Dashboard")
    print("─" * 45)

    print("\n📦 Checking dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"), "-q"],
            check=True,
        )
    except subprocess.CalledProcessError:
        print("⚠️  Some packages may have failed to install — trying to start anyway.")

    print("🌐 Starting dashboard server...")
    env = os.environ.copy()
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            str(ROOT / "dashboard" / "app.py"),
            f"--server.port={PORT}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ],
        env=env,
        cwd=str(ROOT),
    )

    # Wait for Streamlit to be ready
    time.sleep(4)

    print(f"\n✅ Dashboard ready!  →  {URL}")
    print("\nPress Ctrl+C to stop.\n")
    webbrowser.open(URL)

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Sodil. Goodbye!")
        proc.terminate()


if __name__ == "__main__":
    main()
