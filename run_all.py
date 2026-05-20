"""
run_all.py — Self-Contained Streamlit Application Pipeline Runner
===================================================================
Orchestrates and launches:
  1. ML Retraining Pipeline (models/pipeline.py) -> hyperparameter tuning & SHAP plotting
  2. Standalone Streamlit App (streamlit_app.py on port 8501) -> direct in-process RAG & ML
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
    print("opsFlow - Self-Contained Deployment Orchestrator")
    print("=" * 65)
    
    # Step 1: Run Machine Learning Pipeline to train, plot and serialize
    print("\n[Step 1/2] Triggering ML training & diagnostic visualization pipeline...")
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

    # Step 3: Boot Streamlit Operator Interface directly
    print("\n[Step 2/2] Spinning up Streamlit Operations Console on port 8501...")
    frontend_process = subprocess.Popen(
        [str(venv_python), "-m", "streamlit", "run", "streamlit_app.py",
         "--server.port", "8501",
         "--server.address", "0.0.0.0",
         "--browser.gatherUsageStats", "false"],
        cwd=str(base_dir),
        env=os.environ.copy()
    )
    
    # Wait for streamlit server to launch
    time.sleep(3)
    
    print("\n" + "=" * 65)
    print("✅ Self-Contained Streamlit System Deployment Successful!")
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
        # Graceful shutdown of frontend process
        frontend_process.terminate()
        frontend_process.wait()
        print("All processes cleaned up. Have a great day!")

if __name__ == "__main__":
    main()
