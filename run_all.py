"""
run_all.py — Full Pipeline Runner + UI Launcher
=================================================
Runs the complete opsFlow system in sequence:
  1. Task 3 v3: Trains ML models, saves model_summary.json
  2. Integration: Copies model_summary.json into Task 4 knowledge base
  3. Task 4 v3: Runs 5 automated demo queries
  4. Streamlit UI: Launches interactive web interface at localhost:8501

USE THIS WHEN:
  - You want to run the full system in one command
  - You want the interactive web UI for asking your own questions

STOPPING:
  - Press Ctrl+C once to stop the Streamlit server and exit cleanly

Usage:
  cd opsFlow/
  python run_all.py
"""

import os
import sys
import time
import shutil
import subprocess
import webbrowser

def load_env_file():
    # Search for .env file in the current directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'").strip('"')
                    if k:
                        os.environ[k] = v

def main():
    load_env_file()
    print("=" * 60)
    print("opsFlow — AI Maintenance Intelligence System")
    print("=" * 60)
    
    # 1. Determine directories relative to run_all.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    task3_v3_dir = os.path.join(base_dir, "task3", "v3")
    task4_v3_dir = os.path.join(base_dir, "task4", "v3")
    task4_v2_dir = os.path.join(base_dir, "task4", "v2")
    task4_v1_dir = os.path.join(base_dir, "task4", "v1")
    
    # 2. Part 1: Run Task 3 v3 ML Training
    print("\n[1/3] Running Task 3: Equipment Failure Prediction...")
    task3_process = subprocess.run(
        [sys.executable, "main.py"],
        cwd=task3_v3_dir,
        env=os.environ.copy()
    )
    
    if task3_process.returncode != 0:
        print("\n[ERROR] Task 3 ML Training failed. Aborting RAG execution.")
        sys.exit(task3_process.returncode)
        
    # 3. Copy model_summary.json into Task 4 v3 and v2 docs/
    model_summary_src  = os.path.join(task3_v3_dir, "outputs", "model_summary.json")
    model_summary_v2   = os.path.join(base_dir, "task4", "v2", "docs", "model_summary.json")
    model_summary_v3   = os.path.join(base_dir, "task4", "v3", "docs", "model_summary.json")

    if os.path.exists(model_summary_src):
        # Ensure destination directories exist before copying
        os.makedirs(os.path.dirname(model_summary_v2), exist_ok=True)
        os.makedirs(os.path.dirname(model_summary_v3), exist_ok=True)
        shutil.copy(model_summary_src, model_summary_v2)
        print(f"[INTEGRATION] Copied model_summary.json → task4/v2/docs/")
        shutil.copy(model_summary_src, model_summary_v3)
        print(f"[INTEGRATION] Copied model_summary.json → task4/v3/docs/")
    else:
        print(f"[WARNING] model_summary.json not found at {model_summary_src} — skipping copy")
        
    # 4. Part 2: Run Task 4 v3 RAG Assistant (Non-interactive automated queries)
    print("\n[2/3] Running Task 4: RAG Maintenance Assistant (v3 Hybrid Search)...")
    task4_process = subprocess.run(
        [sys.executable, "main.py"],
        cwd=os.path.join(base_dir, "task4", "v3"),
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    if task4_process.returncode != 0:
        print("\n[ERROR] Task 4 RAG Assistant failed.")
        sys.exit(task4_process.returncode)
        
    # 5. Part 3: Launch Streamlit Web UI
    print("\n" + "=" * 60)
    print("opsFlow — Launching Streamlit UI...")
    print("=" * 60)

    try:
        streamlit_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
             "--server.headless", "true",
             "--browser.gatherUsageStats", "false",
             "--server.port", "8501"],
            cwd=base_dir,
            env=os.environ.copy()
        )
        
        # Wait for Streamlit to fully start before opening browser
        print("Starting Streamlit server", end="", flush=True)
        for _ in range(5):
            time.sleep(1)
            print(".", end="", flush=True)
        print(" Ready!")
        
        print("\n✅ Streamlit UI is running at: http://localhost:8501")
        print("Press Ctrl+C to stop the server.\n")
        
        # Open browser automatically after server is ready
        webbrowser.open("http://localhost:8501")
        
        # Keep run_all.py alive until user presses Ctrl+C
        streamlit_process.wait()

    except KeyboardInterrupt:
        print("\n[run_all.py] Stopping Streamlit server...")
        streamlit_process.terminate()
        streamlit_process.wait()
        print("[run_all.py] Server stopped. Goodbye!")

if __name__ == "__main__":
    main()
