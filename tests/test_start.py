"""Exercise the supervisor with real child processes, without cameras or models."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from start import Supervisor, environment


def test_environment_preserves_plugin_and_library_paths(monkeypatch):
    monkeypatch.setenv("GST_PLUGIN_PATH", "/extra/plugins")
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "/extra/lib")
    env = environment()
    assert env["GST_PLUGIN_PATH"].split(os.pathsep) == [str(ROOT / "plugins/build"), "/extra/plugins"]
    assert env["DYLD_FALLBACK_LIBRARY_PATH"] == "/opt/homebrew/lib:/extra/lib"
    assert env["VAULT_TOKEN"] == env["VAULT_DEV_ROOT_TOKEN_ID"]
    assert env["VAULT_MOUNT_POINT"] == "cryptavision"


def test_shutdown_reaps_child_and_descendant(tmp_path):
    supervisor = Supervisor(environment())
    marker = tmp_path / "grandchild"
    code = """
import subprocess, sys, signal
p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
open(sys.argv[1], 'w').write(str(p.pid))
def stop(*_):
    p.wait()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
p.wait()
"""
    process = supervisor.start("ui", [sys.executable, "-c", code, str(marker)])
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists()
    finally:
        supervisor.shutdown()
    assert process.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)


def test_service_failure_detected_and_sibling_stopped():
    supervisor = Supervisor(environment())
    sibling = supervisor.start("vault", [sys.executable, "-c", "import time; time.sleep(60)"])
    failed = supervisor.start("api", [sys.executable, "-c", "raise SystemExit(7)"])
    try:
        failed.wait(timeout=5)
        with pytest.raises(RuntimeError, match="api exited with status 7"):
            supervisor.wait_ready("backend", lambda: False, 10)
    finally:
        supervisor.shutdown()
    assert sibling.poll() is not None


def test_readiness_timeout_stops_services():
    supervisor = Supervisor(environment())
    process = supervisor.start("vault", [sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with pytest.raises(RuntimeError, match="Timed out waiting"):
            supervisor.wait_ready("backend", lambda: False, 0.1)
    finally:
        supervisor.shutdown()
    assert process.poll() is not None


@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM])
def test_signal_during_startup_cleans_children(tmp_path, number):
    marker = tmp_path / "child"
    code = """
import signal, sys
from start import Supervisor, environment
s = Supervisor(environment())
for number in (signal.SIGINT, signal.SIGTERM):
    signal.signal(number, s.on_signal)
try:
    p = s.start('vault', [sys.executable, '-c', 'import time; time.sleep(60)'])
    open(sys.argv[1], 'w').write(str(p.pid))
    s.wait_ready('never ready', lambda: False, 60)
except InterruptedError:
    pass
finally:
    s.shutdown()
"""
    harness = subprocess.Popen([sys.executable, "-c", code, str(marker)], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists()
        harness.send_signal(number)
        output, _ = harness.communicate(timeout=12)
        assert harness.returncode == 0, output.decode()
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
    finally:
        if harness.poll() is None:
            harness.kill()
            harness.wait()
