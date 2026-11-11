#!/usr/bin/env python3
"""Single-terminal local development supervisor (Python standard library only)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
VENV = ROOT / "venv"
BUILD = ROOT / "plugins" / "build"


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(VENV)
    env["PATH"] = os.pathsep.join((str(VENV / "bin"), "/opt/homebrew/bin", env.get("PATH", "")))
    for key, prefix in (("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib"),
                        ("GST_PLUGIN_PATH", str(BUILD))):
        env[key] = os.pathsep.join(filter(None, (prefix, env.get(key))))
    env["PYTHONUNBUFFERED"] = "1"
    # This launcher always owns an isolated, loopback-only development Vault.
    for key in ("VAULT_NAMESPACE", "VAULT_AGENT_ADDR"):
        env.pop(key, None)
    env.update(VAULT_ADDR="http://127.0.0.1:8200", VAULT_TOKEN=secrets.token_hex(24),
               VAULT_MOUNT_POINT="cryptavision", HOST="127.0.0.1", PORT="8000",
               TCP_INGEST_PORT="5555", NEXT_TELEMETRY_DISABLED="1")
    env["VAULT_DEV_ROOT_TOKEN_ID"] = env["VAULT_TOKEN"]
    return env


class Supervisor:
    def __init__(self, env: dict[str, str]):
        self.env = env
        self.children: list[tuple[str, subprocess.Popen, threading.Thread]] = []
        self.stopping = threading.Event()
        self.signal_number = 0
        self.lock = threading.Lock()

    def log(self, name: str, message: str) -> None:
        colors = {"vault": 35, "api": 36, "ui": 34, "edge": 32, "setup": 33}
        label = f"[{name:>5}]"
        if sys.stdout.isatty() and "NO_COLOR" not in self.env:
            label = f"\033[{colors.get(name, 37)}m{label}\033[0m"
        with self.lock:
            print(f"{label} {message.replace(self.env['VAULT_TOKEN'], '[redacted]')}", flush=True)

    def on_signal(self, number, _frame) -> None:
        self.signal_number = number
        self.stopping.set()

    def start(self, name: str, command: list[str], cwd: Path = ROOT) -> subprocess.Popen:
        if self.stopping.is_set():
            raise InterruptedError("Startup interrupted")
        # Each service owns a process group, including npm's Node descendants.
        # Signal handlers only set an event, so spawn/registration finishes even
        # when Ctrl+C arrives during Popen.
        process = subprocess.Popen(command, cwd=cwd, env=self.env,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace",
                                   start_new_session=True)
        reader = threading.Thread(target=self._read, args=(name, process), daemon=True)
        self.children.append((name, process, reader))
        reader.start()
        return process

    def _read(self, name: str, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                self.log(name, line.rstrip())

    def run(self, command: list[str], cwd: Path = ROOT) -> None:
        process = self.start("setup", command, cwd)
        while process.poll() is None:
            if self.stopping.wait(0.1):
                raise InterruptedError("Setup interrupted")
        if process.returncode:
            raise RuntimeError(f"Setup command failed ({process.returncode}): {command[0]}")
        # Completed one-shot commands no longer own a service group. Forget
        # their IDs so a long-running session never signals a reused group ID.
        entry = next(entry for entry in self.children if entry[1] is process)
        entry[2].join(timeout=1)
        self.children.remove(entry)

    def check_children(self) -> None:
        if self.stopping.is_set():
            raise InterruptedError("Shutdown requested")
        for name, child, _ in self.children:
            if name != "setup" and child.poll() is not None:
                raise RuntimeError(f"{name} exited with status {child.returncode}; stopping the stack")

    def wait_ready(self, description: str, probe, timeout: float) -> None:
        self.log("start", f"Waiting for {description}...")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.check_children()
            if probe():
                self.check_children()
                return
            self.stopping.wait(0.2)
        raise RuntimeError(f"Timed out waiting for {description}")

    @staticmethod
    def _group_exists(group: int) -> bool:
        try:
            os.killpg(group, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # macOS can return EPERM for a process group that has just vanished.
            # Verify group membership instead of aborting the rest of cleanup.
            groups = subprocess.check_output(["ps", "-axo", "pgid="], text=True)
            return str(group) in groups.split()

    @staticmethod
    def _send_group(process: subprocess.Popen, number: int) -> None:
        # Send even if the parent exited: npm may have left descendants alive.
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            pass
        except PermissionError:
            if Supervisor._group_exists(process.pid):
                raise

    def shutdown(self) -> None:
        if not self.children:
            return
        self.log("start", "Stopping all managed processes...")
        errors = []
        def send(name, process, number):
            try:
                self._send_group(process, number)
            except OSError as exc:
                errors.append(f"{name}: {exc}")
        for name, process, _ in reversed(self.children):
            send(name, process, signal.SIGTERM)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            alive = False
            for _, process, _ in self.children:
                process.poll()  # Reap direct children while waiting for their groups.
                if self._group_exists(process.pid):
                    alive = True
            if not alive:
                break
            time.sleep(0.1)
        for name, process, _ in reversed(self.children):
            send(name, process, signal.SIGKILL)
        for name, process, reader in self.children:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                errors.append(f"{name}: process did not exit")
            reader.join(timeout=1)
        if errors:
            raise RuntimeError("Cleanup could not finish: " + "; ".join(errors))
        self.log("start", "All managed processes stopped.")


def http_json(url: str):
    # Ignore corporate proxy settings for loopback readiness probes.
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=1) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return None


def tcp_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def check_ports() -> None:
    for port in (8200, 8201, 8000, 5555, 3000):
        with socket.socket() as sock:
            # Ignore TIME_WAIT from the previous dev run, while rejecting listeners.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except PermissionError as exc:
                raise RuntimeError(f"Cannot check port {port}: loopback networking permission denied") from exc
            except OSError as exc:
                raise RuntimeError(f"Port {port} is unavailable; stop its existing owner before launching") from exc


def setup(supervisor: Supervisor) -> None:
    python = str(VENV / "bin" / "python")
    if not Path(python).exists():
        supervisor.run([sys.executable, "-m", "venv", str(VENV)])
    supervisor.run([python, "-m", "pip", "install", "--only-binary=tenseal",
                    "-r", "requirements-api.txt", "-r", "requirements-crypto.txt",
                    "-r", "requirements-enclave.txt", "-r", "requirements-ai.txt", "PyGObject"])
    supervisor.run(["cmake", "-S", "plugins", "-B", str(BUILD)])
    supervisor.run(["cmake", "--build", str(BUILD), "--parallel", "2"])
    supervisor.run(["npm", "ci"], ROOT / "frontend")


def preflight(supervisor: Supervisor) -> None:
    # Run imports in a fresh interpreter so macOS sees DYLD_* at process startup.
    supervisor.run([str(VENV / "bin" / "python"), "-c", """
import sys
assert sys.version_info[:2] == (3, 11) and sys.prefix != sys.base_prefix, 'Python 3.11 venv required'
import fastapi, uvicorn, hvac, tenseal, torch, torchvision, numpy, cv2, PIL, gi
import ultralytics, transformers, facenet_pytorch
from transformers import CLIPModel, CLIPProcessor
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)
for name in ('crypta-frame-filter', 'cryptanetworksink', 'videoconvert', 'appsink', 'appsrc'):
    assert Gst.ElementFactory.find(name), 'Missing GStreamer element: ' + name
print('Python 3.11, AI, crypto and native GStreamer plugins verified.')
"""
    ])
    if not (ROOT / "frontend/node_modules/next/package.json").exists():
        raise RuntimeError("Dashboard dependencies missing; run ./start_cryptavision.sh --setup")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="test_surveillance_feed.mp4", help="video path (default: test_surveillance_feed.mp4), webcam, RTSP URI, or test")
    parser.add_argument("--setup", action="store_true", help="Install Python/frontend dependencies and build plugins, then start")
    parser.add_argument("--check", action="store_true", help="Validate environment and free ports without starting services")
    parser.add_argument("--timeout", type=float, default=120, help="Service readiness timeout in seconds")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    # Shell activation is optional; always select this project's own venv.
    if VENV.exists() and Path(sys.prefix) != VENV:
        python = VENV / "bin/python"
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
    if sys.version_info[:2] != (3, 11):
        parser.error("Use Python 3.11: /opt/homebrew/bin/python3.11 start.py --setup")
    if not (VENV / "bin/python").exists() and not args.setup:
        parser.error("Virtual environment missing; run ./start_cryptavision.sh --setup")

    supervisor = Supervisor(environment())
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, supervisor.on_signal)
    result = 0
    try:
        for executable in ("vault", "npm", "node", *( ("cmake", "pkg-config") if args.setup else () )):
            if not shutil.which(executable, path=supervisor.env["PATH"]):
                raise RuntimeError(f"Required executable missing from PATH: {executable}")
        if args.setup:
            setup(supervisor)
        check_ports()
        preflight(supervisor)
        if args.check:
            supervisor.log("start", "Preflight passed. Ports 8200, 8201, 8000, 5555 and 3000 are free.")
            return 0
        if args.source == "test_surveillance_feed.mp4" and not (ROOT / args.source).is_file():
            raise RuntimeError("Sample video missing; run ./download_sample_video.sh first (or use --source webcam)")
        supervisor.start("vault", ["vault", "server", "-dev", "-dev-no-store-token",
                                  "-dev-listen-address=127.0.0.1:8200"])
        # Passed via environment instead of a command-line token or ~/.vault-token.
        # Vault's dev server reads VAULT_DEV_ROOT_TOKEN_ID.
        supervisor.wait_ready("Vault", lambda: bool(http_json(supervisor.env["VAULT_ADDR"] + "/v1/sys/health")), args.timeout)
        supervisor.run(["vault", "secrets", "enable", "-path=cryptavision", "kv-v2"])
        supervisor.start("api", [str(VENV / "bin/python"), "main.py"])
        def backend_ready():
            health = http_json("http://127.0.0.1:8000/health")
            return bool(health and health.get("vault_status") == "connected" and tcp_ready(5555))
        supervisor.wait_ready("API, Vault connectivity and TCP ingest", backend_ready, args.timeout)
        supervisor.start("ui", ["npm", "run", "dev", "--", "--hostname", "127.0.0.1", "--port", "3000"], ROOT / "frontend")
        supervisor.wait_ready("dashboard on port 3000", lambda: tcp_ready(3000), args.timeout)
        supervisor.start("edge", [str(VENV / "bin/python"), "edge_node/local_pipeline.py", "--source", args.source])
        supervisor.log("start", "Services launched. Dashboard: http://localhost:3000 | API: http://localhost:8000")
        supervisor.log("start", "Edge model loading may take time on first run. Press Ctrl+C to stop everything.")
        while True:
            supervisor.check_children()
            supervisor.stopping.wait(0.2)
    except InterruptedError:
        result = 128 + (supervisor.signal_number or signal.SIGINT)
    except (OSError, RuntimeError) as exc:
        supervisor.log("error", str(exc))
        result = 1
    finally:
        try:
            supervisor.shutdown()
        except (OSError, RuntimeError) as exc:
            supervisor.log("error", str(exc))
            result = 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
