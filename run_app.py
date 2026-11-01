#!/usr/bin/env python3
"""
Launcher for IoT Intrusion Detection Live Flow Inspector.
Serves both the FastAPI backend and the compiled React frontend on http://127.0.0.1:8000.
"""

import sys
import os
import subprocess
from pathlib import Path

# Auto-switch to .venv python if not already running in it
base_dir = Path(__file__).resolve().parent
venv_python = base_dir / ".venv" / "bin" / "python"
if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

# Auto-switch to .venv python if not already running in it
base_dir = Path(__file__).resolve().parent
venv_dir = base_dir / ".venv"
venv_python = venv_dir / "bin" / "python"

in_venv = (sys.prefix != sys.base_prefix) and (Path(sys.prefix).resolve() == venv_dir.resolve())
if not in_venv and venv_python.exists():
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = str(venv_dir / "bin") + os.pathsep + env.get("PATH", "")
    os.execve(str(venv_python), [str(venv_python)] + sys.argv, env)

def main():
    dist_dir = base_dir / "frontend" / "dist"

    if not dist_dir.exists():
        print("Frontend build not found. Building frontend bundle...")
        subprocess.run(["npm", "run", "build", "--prefix", "frontend"], check=True)

    print("  🚀 IoT IDS LIVE FLOW INSPECTOR & ANOMALY DETECTOR")
    print("  Dataset: CICIoT2023 | Models: DT (Top-15 / Best), RF, XGBoost")
    print("  URL: http://127.0.0.1:8000")

    try:
        import uvicorn
    except ImportError:
        print("\n❌ Error: 'uvicorn' is not installed in the virtual environment.")
        print("Run the following to install all dependencies:")
        print("  source .venv/bin/activate && pip install -r requirements.txt\n")
        sys.exit(1)

    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
