"""One-command local launcher for Comeback AI."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
INSTALL_MARKER = VENV_DIR / ".comeback-install"


def project_fingerprint() -> str:
    return hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()


def prepare_environment(force: bool = False) -> None:
    if sys.version_info < (3, 11):  # noqa: UP036 - this script runs before installation
        raise SystemExit("Comeback AI requires Python 3.11 or newer.")

    if not PYTHON.exists():
        print("[setup] Creating .venv ...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)

    fingerprint = project_fingerprint()
    installed = INSTALL_MARKER.read_text(encoding="utf-8") if INSTALL_MARKER.exists() else ""
    if force or installed != fingerprint:
        print("[setup] Installing the project and development tools ...")
        subprocess.run(
            [str(PYTHON), "-m", "pip", "install", "-e", ".[ui,dev]"],
            cwd=ROOT,
            check=True,
        )
        INSTALL_MARKER.write_text(fingerprint, encoding="utf-8")


def train_if_needed(retrain: bool = False) -> None:
    model = ROOT / "artifacts" / "risk_model.joblib"
    if retrain or not model.exists():
        print("[ml] Training and evaluating candidate models ...")
        subprocess.run([str(PYTHON), "-m", "comeback_ai.ml.train"], cwd=ROOT, check=True)


def wait_for_api(process: subprocess.Popen, timeout: int = 45) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("The API exited before becoming ready.")
        try:
            with urllib.request.urlopen("http://localhost:8000/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    raise TimeoutError("The API did not become ready within 45 seconds.")


def stop(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def serve(open_browser: bool = True) -> None:
    environment = os.environ.copy()
    environment["API_URL"] = "http://localhost:8000"
    processes: list[subprocess.Popen] = []
    try:
        api = subprocess.Popen(
            [str(PYTHON), "-m", "uvicorn", "comeback_ai.api.main:app", "--port", "8000"],
            cwd=ROOT,
            env=environment,
        )
        processes.append(api)
        wait_for_api(api)

        ui = subprocess.Popen(
            [
                str(PYTHON),
                "-m",
                "streamlit",
                "run",
                "ui/app.py",
                "--server.port=8501",
                "--server.headless=true",
            ],
            cwd=ROOT,
            env=environment,
        )
        processes.append(ui)
        time.sleep(1)
        if ui.poll() is not None:
            raise RuntimeError("The frontend exited before becoming ready.")

        print("\nComeback AI is ready:")
        print("  App:      http://localhost:8501")
        print("  API docs: http://localhost:8000/docs")
        print("  Stop:     Ctrl+C\n")
        if open_browser:
            webbrowser.open("http://localhost:8501")

        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping Comeback AI ...")
    finally:
        stop(processes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up and run Comeback AI locally.")
    parser.add_argument("--setup", action="store_true", help="reinstall project dependencies")
    parser.add_argument("--retrain", action="store_true", help="retrain the ML model")
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open the app automatically"
    )
    args = parser.parse_args()

    prepare_environment(force=args.setup)
    train_if_needed(retrain=args.retrain)
    serve(open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
