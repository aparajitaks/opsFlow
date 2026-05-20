"""
run_all.py — Production-Grade Pipeline Runner + Service Orchestrator
===================================================================
Orchestrates and launches:
  1. ML Retraining Pipeline (models/pipeline.py) -> hyperparameter tuning & SHAP plotting
  2. FastAPI Uvicorn Server (api/main.py on port 8000) -> lifespan warmups, DB builds
  3. Streamlit Operator Console (frontend/app.py on port 8501) -> interactive operations UI
"""

import os
import sys
import time
import shutil
import subprocess
import webbrowser
from pathlib import Path

def load_env_file(base_dir: Path):
    """Loads env variables from root .env if present."""
    env_path = base_dir / ".env"
    if env_path.exists():
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
    base_dir = Path(__file__).resolve().parent
    load_env_file(base_dir)
    
    # Determine the python binary inside the virtual environment
    venv_python = base_dir / "venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = Path(sys.executable) # Fallback to active runner binary
        
    print("=" * 65)
    print("opsFlow - Production-Grade Deployment Orchestrator")
    print("=" * 65)
    
    # Step 1: Run Machine Learning Pipeline to train, plot and serialize
    print("\n[Step 1/3] Triggering ML training & diagnostic visualization pipeline...")
    ml_run = subprocess.run(
        [str(venv_python), "-m", "models.pipeline"],
        cwd=str(base_dir),
        env=os.environ.copy()
    )
    if ml_run.returncode != 0:
        print("[ERROR] Machine learning pipeline failed. Halting startup.")
        sys.exit(ml_run.returncode)
        
    # Step 2: Ensure model_summary.json is in docs/ directory for RAG indexing
    summary_src = base_dir / "models" / "artifacts" / "model_summary.json"
    docs_dir = base_dir / "docs"
    docs_dir.mkdir(exist_ok=True)
    summary_dest = docs_dir / "model_summary.json"
    
    if summary_src.exists():
        shutil.copy(summary_src, summary_dest)
        print(f"[Integration] Synced model_summary.json -> {summary_dest}")
    else:
        print(f"[Warning] model_summary.json not found at {summary_src}. Proceeding with legacy/default configs.")

    # Step 3: Boot FastAPI REST Service
    print("\n[Step 2/3] Spinning up FastAPI Backend Service on port 8000...")
    backend_process = subprocess.Popen(
        [str(venv_python), "-m", "api.main"],
        cwd=str(base_dir),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for backend to print startup message or warm models
    print("Initializing backend and warming retrieval models", end="", flush=True)
    time.sleep(5)
    print(" Ready!")
    
    # Step 4: Boot Streamlit Operator Interface
    print("\n[Step 3/3] Spinning up Streamlit Operations Console on port 8501...")
    frontend_process = subprocess.Popen(
        [str(venv_python), "-m", "streamlit", "run", "frontend/app.py",
         "--server.port", "8501",
         "--server.address", "0.0.0.0",
         "--browser.gatherUsageStats", "false"],
        cwd=str(base_dir),
        env=os.environ.copy()
    )
    
    # Wait for streamlit server to launch
    time.sleep(3)
    
    print("\n" + "=" * 65)
    print("✅ System Deployment Successful!")
    print("FastAPI Backend:     http://localhost:8000")
    print("Interactive Web UI:  http://localhost:8501")
    print("Press Ctrl+C to terminate services cleanly.")
    print("=" * 65 + "\n")
    
    webbrowser.open("http://localhost:8501")
    
    try:
        # Keep process running
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down opsFlow services...")
    finally:
        # Graceful shutdown of backend and frontend processes
        frontend_process.terminate()
        backend_process.terminate()
        frontend_process.wait()
        backend_process.wait()
        print("All processes cleaned up. Have a great day!")

if __name__ == "__main__":
    main()
