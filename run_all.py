# run_all.py
# Runs Task 3 v3 first (trains models, saves model_summary.json)
# Then runs Task 4 v2 (loads knowledge base including model_summary, starts RAG assistant)

import os
import sys
import shutil
import subprocess

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
    task4_v2_dir = os.path.join(base_dir, "task4", "v2")
    task4_v1_dir = os.path.join(base_dir, "task4", "v1")
    
    # 2. Part 1: Run Task 3 v3 ML Training
    print("\n[1/2] Running Task 3: Equipment Failure Prediction...")
    task3_process = subprocess.run(
        [sys.executable, "main.py"],
        cwd=task3_v3_dir,
        env=os.environ.copy()
    )
    
    if task3_process.returncode != 0:
        print("\n[ERROR] Task 3 ML Training failed. Aborting RAG execution.")
        sys.exit(task3_process.returncode)
        
    # 3. Copy model_summary.json into Task 4 v2 and v1 docs/
    model_summary_src = os.path.join(task3_v3_dir, "outputs", "model_summary.json")
    if not os.path.exists(model_summary_src):
        print(f"\n[ERROR] Could not locate model summary JSON at: {model_summary_src}")
        sys.exit(1)
        
    # Destination for v2
    v2_docs_dir = os.path.join(task4_v2_dir, "docs")
    os.makedirs(v2_docs_dir, exist_ok=True)
    model_summary_dest_v2 = os.path.join(v2_docs_dir, "model_summary.json")
    shutil.copy(model_summary_src, model_summary_dest_v2)
    print(f"\n[INTEGRATION] Successfully copied model_summary.json to Task 4 v2 docs: {model_summary_dest_v2}")
    
    # Destination for v1 (optional helper)
    v1_docs_dir = os.path.join(task4_v1_dir, "docs")
    os.makedirs(v1_docs_dir, exist_ok=True)
    model_summary_dest_v1 = os.path.join(v1_docs_dir, "model_summary.json")
    shutil.copy(model_summary_src, model_summary_dest_v1)
    
    # 4. Part 2: Run Task 4 v2 RAG Assistant
    print("\n[2/2] Running Task 4: RAG Maintenance Assistant...")
    # Ensure subprocess inherits standard input for interactive prompt looping
    task4_process = subprocess.run(
        [sys.executable, "main.py"],
        cwd=task4_v2_dir,
        env=os.environ.copy(),
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    if task4_process.returncode != 0:
        print("\n[ERROR] Task 4 RAG Assistant failed.")
        sys.exit(task4_process.returncode)
        
    print("\n" + "=" * 60)
    print("opsFlow execution completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
