"""Real GStreamer decoding/segment-loop regression, with inference stubbed out."""
import json
import socket
import struct
import threading
import shutil
import subprocess
import time
from unittest.mock import MagicMock

import pytest

from edge_node.local_pipeline import LocalEdgePipeline, Gst, GST_AVAILABLE


@pytest.mark.skipif(not GST_AVAILABLE or not shutil.which("ffmpeg"), reason="GStreamer and ffmpeg required")
def test_local_video_repeats_without_restarting_pipeline(tmp_path, monkeypatch):
    video = tmp_path / "walking test.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=320x240:rate=30", "-t", "0.6", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    monkeypatch.setattr(LocalEdgePipeline, "_init_engines", lambda self: None)
    # Exercise the real native sink: replacing it with an async=false fakesink
    # previously hid a startup deadlock in the production pipeline.
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    server.settimeout(10)
    packets = []
    failures = []
    def receive():
        try:
            conn, _ = server.accept()
            with conn, conn.makefile("rb") as stream:
                conn.settimeout(10)
                while True:
                    header = stream.read(9)
                    if not header:
                        break
                    magic, size, kind = struct.unpack("!IIB", header)
                    assert magic == 0x43525950 and kind == 1
                    packets.append(json.loads(stream.read(size)))
        except Exception as exc:
            failures.append(exc)
    receiver = threading.Thread(target=receive, daemon=True)
    receiver.start()
    edge = LocalEdgePipeline(source_uri=str(video), cloud_port=server.getsockname()[1])
    edge.ai_engine = MagicMock()
    edge.ai_engine.semantic_analyzer.process_frame.return_value = {"status": "ok"}
    edge.ai_engine.latent_store.drain.return_value = []
    try:
        edge.start()
        pipeline = edge._pipeline
        deadline = time.monotonic() + 8
        while edge.loop_count < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert edge.loop_count >= 3
        assert len(packets) > 18
        assert edge.frames_in > 18  # More frames than the original 0.6-second file.
        assert edge._pipeline is pipeline
        assert not edge._eos_event.is_set()
        # Playback should remain near real time, rather than decoding an entire
        # clip as fast as possible and flooding the dashboard.
        assert edge.frames_in < 180
    finally:
        edge.stop()
        server.close()
        receiver.join(timeout=3)
    assert not failures
