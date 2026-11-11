#!/usr/bin/env python3
"""
CryptaVision — Hardware-Agnostic Local Edge Video Analytics Pipeline
=====================================================================
Universal edge runner designed for local execution on developer PCs, Macs,
and commodity servers without requiring NVIDIA Jetson-specific hardware.

Architectural Model:
--------------------
1. Dynamic Video Ingest (Hardware-Agnostic):
   - Supports local webcams ('webcam', '0'), RTSP streams ('rtsp://...'),
     HTTP/HLS streams ('http://...'), and local video files (*.mp4, *.mkv).
   - Dynamically resolves the GStreamer source string via `get_source_element()`.
   - Utilizes standard software decoding (avdec_h264 / uridecodebin) and standard
     `videoconvert` for BGRx colorspace negotiation.
   - Zero-copy `tee` branching into decoupled queues.

2. On-Device AI & Homomorphic Encryption (Zero Raw Video Egress):
   - Stream A (appsink 'sink_a'): YOLOv8 pose keypoints & CLIP FP16 embeddings.
   - Stream B (appsink 'sink_b'): Face detection, VAE latent extraction ([1, 128] FP32),
     and on-edge TenSEAL CKKS homomorphic encryption (N=8192, scale=2^40).
   - Zero raw or obfuscated video frames exit over the WAN.

3. Low-Level C++ Network Sink Hand-off:
   - Serialized telemetry packets (Type 0x01 JSON, Type 0x02 CKKS) are pushed
     into an internal `appsrc` ('src_b' / 'tx_appsrc') linked directly to our
     custom C++ `cryptanetworksink`.
   - The C++ sink writes directly to raw POSIX TCP sockets with non-blocking I/O.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = "test_surveillance_feed.mp4"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# GStreamer Python bindings
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import GLib, Gst
    GST_AVAILABLE = True
except (ImportError, ValueError):
    gi = None  # type: ignore
    GLib = None  # type: ignore
    Gst = None  # type: ignore
    GST_AVAILABLE = False

# Import Phase 1, 2, and 3 Core Engines
try:
    from cryptavision_pipeline import MappedBuffer, FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE
except (ImportError, ModuleNotFoundError):
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FRAME_RATE = 30
    MappedBuffer = None  # type: ignore

try:
    from ai_engine import CryptaAIEngine
except (ImportError, ModuleNotFoundError):
    CryptaAIEngine = None  # type: ignore

from crypto_engine import CryptoEngine, TENSEAL_AVAILABLE

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [LocalEdge]: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("LocalEdge")


# ══════════════════════════════════════════════════════════════════════════════
#  Dynamic Source Resolution Helper
# ══════════════════════════════════════════════════════════════════════════════
def get_source_element(source_arg: str) -> str:
    """
    Dynamically resolves the GStreamer source element string based on input type.

    Rules:
      1. If `source_arg` is '0' or 'webcam' (case-insensitive), returns "autovideosrc".
      2. If it starts with 'rtsp://' or 'http://' (or 'https://'), returns "uridecodebin uri=" + source_arg.
      3. If it is a valid local file path (checked via os.path.isfile), resolves the absolute
         path and returns "uridecodebin uri=file://" + abs_path.
      4. Fallback: returns "uridecodebin uri=" + source_arg (or "videotestsrc" for synthetic tests).

    Args:
        source_arg: String specifying the webcam index, streaming URI, or local file path.

    Returns:
        str: Valid GStreamer source string ready for pipeline injection.
    """
    clean_arg = str(source_arg).strip()
    lower_arg = clean_arg.lower()

    # Rule 1: Live local webcam
    if lower_arg in ("0", "webcam") or lower_arg == "/dev/video0":
        return "autovideosrc"

    # Rule 2: Remote RTSP or HTTP/HTTPS stream
    if lower_arg.startswith(("rtsp://", "http://", "https://")):
        return f"uridecodebin uri={json.dumps(clean_arg)}"

    # Rule 3: Valid local video file
    if os.path.isfile(clean_arg):
        return f"uridecodebin uri={json.dumps(Path(clean_arg).resolve().as_uri())}"

    # Additional standard URI or synthetic fallbacks
    if lower_arg.startswith("file://"):
        return f"uridecodebin uri={json.dumps(clean_arg)}"
    if lower_arg in ("videotestsrc", "test"):
        return "videotestsrc"

    return f"uridecodebin uri={json.dumps(clean_arg)}"


# ══════════════════════════════════════════════════════════════════════════════
#  Hardware-Agnostic Local Edge Pipeline
# ══════════════════════════════════════════════════════════════════════════════
class LocalEdgePipeline:
    """
    Universal hardware-agnostic edge video analytics pipeline.
    Runs on standard PCs/Macs using software decoding, dynamic source resolution,
    and streams encrypted telemetry through the custom C++ `cryptanetworksink`.
    """

    def __init__(
        self,
        source_uri: str = DEFAULT_SOURCE,
        cloud_host: str = "127.0.0.1",
        cloud_port: int = 5555,
        process_interval: int = 5,
        latent_dim: int = 128,
    ):
        if source_uri == DEFAULT_SOURCE:
            source_uri = str(PROJECT_ROOT / DEFAULT_SOURCE)
            if not Path(source_uri).is_file():
                raise FileNotFoundError("Sample video missing; run ./download_sample_video.sh or pass --source webcam")
        self.source_uri = source_uri
        self._loop_video = Path(source_uri).is_file() or source_uri.lower().startswith("file://")
        self._segment_started = False
        self._stopping = False
        self.loop_count = 0
        self._video_source = None
        self.cloud_host = cloud_host
        self.cloud_port = cloud_port
        self.process_interval = process_interval
        self.latent_dim = latent_dim

        self._pipeline: Optional[Gst.Pipeline] = None
        self._loop: Optional[GLib.MainLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._eos_event = threading.Event()
        self._running = False

        # Internal appsrc handle ('src_b' / 'tx_appsrc') for passing telemetry to cryptanetworksink
        self._tx_appsrc: Optional[Gst.Element] = None
        self._tx_lock = threading.Lock()

        # Telemetry metrics
        self.frames_in = 0
        self.stream_a_packets = 0
        self.stream_b_packets = 0

        # Initialize GStreamer & Core AI/Crypto Engines
        if GST_AVAILABLE and Gst is not None:
            Gst.init(None)
        self._init_engines()

    def _init_engines(self) -> None:
        """Instantiates Edge AI engine (YOLOv8 + CLIP + FaceVAE) and TenSEAL CKKS crypto engine."""
        log.info("Initializing Local Edge AI Engine (YOLOv8 + CLIP + FaceVAE)...")
        self.ai_engine = CryptaAIEngine(
            process_interval=self.process_interval,
            latent_dim=self.latent_dim,
        )

        log.info("Initializing On-Edge TenSEAL CKKS Homomorphic Encryption Engine...")
        if not TENSEAL_AVAILABLE:
            raise RuntimeError(
                "TenSEAL is required on edge device. Install via pip install tenseal>=0.3.14"
            )
        self.crypto_engine = CryptoEngine(latent_dim=self.latent_dim)
        log.info("CKKS Engine Ready (N=8192, [60,40,40,60], scale=2^40).")

    def _build_pipeline(self) -> Gst.Pipeline:
        """
        Builds the hardware-agnostic GStreamer pipeline graph.
        Dynamically resolves the video source, converts to BGRx, forks via a zero-copy
        tee into sink_a and sink_b, and links src_b to cryptanetworksink.
        """
        if not GST_AVAILABLE or Gst is None:
            raise RuntimeError(
                "GStreamer Python bindings (gi.repository.Gst) are required to build LocalEdgePipeline. "
                "Ensure PyGObject and GStreamer are installed."
            )

        source_str = get_source_element(self.source_uri)
        if source_str.startswith("uridecodebin "):
            source_str = source_str.replace("uridecodebin ", "uridecodebin name=video_source ", 1)
        log.info("Resolved dynamic video source: '%s' -> [%s]", self.source_uri, source_str)

        # Verify C++ cryptanetworksink plugin availability
        has_net_sink = Gst.ElementFactory.find("cryptanetworksink") is not None
        if not has_net_sink:
            raise RuntimeError(
                "GStreamer element 'cryptanetworksink' not found! "
                "Ensure libcrypta_network_sink is compiled and GST_PLUGIN_PATH is set.\n"
                "Build with: cd plugins && mkdir -p build && cd build && cmake .. && make"
            )

        # ── Pipeline Graph Assembly ───────────────────────────────────────────
        # Ingest routing:
        # 1. uridecodebin: already handles demuxing and decoding internally.
        # 2. autovideosrc / videotestsrc: outputs raw frames directly.
        # 3. other sources: route through decodebin.
        if "uridecodebin" in source_str or "autovideosrc" in source_str or "videotestsrc" in source_str:
            source_pipe = f"{source_str} ! videoconvert ! videoscale ! videorate"
        elif "decodebin" in source_str:
            source_pipe = f"{source_str} ! videoconvert ! videoscale ! videorate"
        else:
            source_pipe = f"{source_str} ! decodebin ! videoconvert ! videoscale ! videorate"

        caps_str = f"video/x-raw,format=BGRx,width={FRAME_WIDTH},height={FRAME_HEIGHT},framerate={FRAME_RATE}/1"

        pipeline_desc = f"""
            {source_pipe} ! {caps_str} ! identity sync=true ! tee name=t
            t. ! queue name=queue_a leaky=downstream max-size-buffers=3 ! appsink name=sink_a emit-signals=true sync=false drop=true max-buffers=1
            t. ! queue name=queue_b leaky=downstream max-size-buffers=3 ! appsink name=sink_b emit-signals=true sync=false drop=true max-buffers=1
            appsrc name=src_b is-live=true format=time block=false ! cryptanetworksink name=net_sink host={self.cloud_host} port={self.cloud_port} timeout-ms=1000 auto-reconnect=true sync=false async=false
        """

        log.info("Constructing GStreamer pipeline via parse_launch...")
        try:
            pipeline = Gst.parse_launch(pipeline_desc)
        except Exception as exc:
            log.warning("Gst.parse_launch encountered an issue (%s); falling back to programmatic build.", exc)
            pipeline = self._build_pipeline_programmatic(source_str, caps_str)

        # ── Dynamic Pad Safeguard for uridecodebin / decodebin ────────────────
        conv_el = pipeline.get_by_name("videoconvert0") or pipeline.get_by_name("conv")
        if conv_el is not None:
            sink_pad = conv_el.get_static_pad("sink")
            # Gst.Iterator is not a Python iterable in all PyGObject releases.
            elements = pipeline.iterate_elements()
            while True:
                status, el = elements.next()
                if status == Gst.IteratorResult.RESYNC:
                    elements.resync()
                    continue
                if status != Gst.IteratorResult.OK:
                    break
                factory = el.get_factory()
                if factory and factory.get_name() in ("uridecodebin", "decodebin"):
                    def _on_pad_added(src_el, pad, target_sink_pad):
                        if not target_sink_pad.is_linked():
                            caps = pad.get_current_caps() or pad.query_caps(None)
                            if caps and caps.get_size() > 0:
                                name = caps.get_structure(0).get_name()
                                if "video" in name:
                                    ret = pad.link(target_sink_pad)
                                    log.info("Dynamically linked video pad '%s' -> sink pad (return: %s)", pad.get_name(), ret)
                    el.connect("pad-added", _on_pad_added, sink_pad)

        # ── Resolve and Wire App Elements ─────────────────────────────────────
        sink_a = pipeline.get_by_name("sink_a")
        sink_b = pipeline.get_by_name("sink_b")
        src_b = pipeline.get_by_name("src_b")

        if sink_a is None or sink_b is None or src_b is None:
            raise RuntimeError("Pipeline failed to initialize sink_a, sink_b, or src_b elements.")

        self._tx_appsrc = src_b
        self._video_source = pipeline.get_by_name("video_source") or pipeline.get_by_name("src")

        # Connect sample listeners
        sink_a.connect("new-sample", self._on_stream_a_sample)
        sink_b.connect("new-sample", self._on_stream_b_sample)

        log.info("Hardware-agnostic pipeline graph successfully constructed.")
        return pipeline

    def _build_pipeline_programmatic(self, source_str: str, caps_str: str) -> Gst.Pipeline:
        """Fallback programmatic builder using discrete Gst.ElementFactory calls."""
        pipeline = Gst.Pipeline.new("local-edge-pipeline")

        # Ingest element
        if source_str == "autovideosrc":
            src = Gst.ElementFactory.make("autovideosrc", "src")
        elif source_str == "videotestsrc":
            src = Gst.ElementFactory.make("videotestsrc", "src")
        elif "uridecodebin" in source_str:
            src = Gst.ElementFactory.make("uridecodebin", "src")
            uri = json.loads(source_str.split("uri=")[-1].strip())
            src.set_property("uri", uri)
        else:
            src = Gst.ElementFactory.make("uridecodebin", "src")
            src.set_property("uri", self.source_uri)

        conv = Gst.ElementFactory.make("videoconvert", "conv")
        scale = Gst.ElementFactory.make("videoscale", "scale")
        rate = Gst.ElementFactory.make("videorate", "rate")
        capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        capsfilter.set_property("caps", Gst.Caps.from_string(caps_str))
        pacing = Gst.ElementFactory.make("identity", "video_pacing")
        pacing.set_property("sync", True)

        tee = Gst.ElementFactory.make("tee", "t")

        # Stream A
        queue_a = Gst.ElementFactory.make("queue", "queue_a")
        queue_a.set_property("leaky", 2)
        queue_a.set_property("max-size-buffers", 3)
        sink_a = Gst.ElementFactory.make("appsink", "sink_a")
        sink_a.set_property("emit-signals", True)
        sink_a.set_property("sync", False)
        sink_a.set_property("drop", True)
        sink_a.set_property("max-buffers", 1)

        # Stream B
        queue_b = Gst.ElementFactory.make("queue", "queue_b")
        queue_b.set_property("leaky", 2)
        queue_b.set_property("max-size-buffers", 3)
        sink_b = Gst.ElementFactory.make("appsink", "sink_b")
        sink_b.set_property("emit-signals", True)
        sink_b.set_property("sync", False)
        sink_b.set_property("drop", True)
        sink_b.set_property("max-buffers", 1)

        # Network Egress
        src_b = Gst.ElementFactory.make("appsrc", "src_b")
        src_b.set_property("is-live", True)
        src_b.set_property("format", Gst.Format.TIME)
        src_b.set_property("block", False)

        net_sink = Gst.ElementFactory.make("cryptanetworksink", "net_sink")
        net_sink.set_property("host", self.cloud_host)
        net_sink.set_property("port", self.cloud_port)
        net_sink.set_property("timeout-ms", 1000)
        net_sink.set_property("auto-reconnect", True)
        net_sink.set_property("sync", False)
        # Metadata cannot exist until video appsinks run. Waiting for preroll on
        # this sink creates a circular dependency with those video appsinks.
        net_sink.set_property("async", False)

        # Add all elements
        for el in [src, conv, scale, rate, capsfilter, pacing, tee, queue_a, sink_a, queue_b, sink_b, src_b, net_sink]:
            pipeline.add(el)

        # Link source
        if src.get_factory().get_name() == "uridecodebin":
            def _on_pad_added(src_el, pad, target):
                sink_pad = target.get_static_pad("sink")
                if not sink_pad.is_linked():
                    pad.link(sink_pad)
            src.connect("pad-added", _on_pad_added, conv)
        else:
            src.link(conv)

        conv.link(scale)
        scale.link(rate)
        rate.link(capsfilter)
        capsfilter.link(pacing)
        pacing.link(tee)

        tee.link(queue_a)
        queue_a.link(sink_a)

        tee.link(queue_b)
        queue_b.link(sink_b)

        src_b.link(net_sink)

        return pipeline

    # ── Hot-Path Callbacks (Streaming Threads) ─────────────────────────────────

    def _on_stream_a_sample(self, appsink: Gst.Element) -> Gst.FlowReturn:
        """
        Stream A: Extracts YOLO pose and CLIP embeddings, serializes to JSON,
        and pushes into cryptanetworksink (Type 0x01).
        """
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        pts_ns = buf.pts
        self.frames_in += 1

        try:
            with MappedBuffer(buf, Gst.MapFlags.READ) as (info, data):
                frame = (
                    np.frombuffer(data, dtype=np.uint8)
                    .reshape(FRAME_HEIGHT, FRAME_WIDTH, 4)
                    .copy()[..., :3]
                )
        except Exception as exc:
            log.warning("Stream A buffer mapping error: %s", exc)
            return Gst.FlowReturn.OK

        # Execute AI inference
        result = self.ai_engine.semantic_analyzer.process_frame(frame, pts_ns)
        if result.get("status") == "skipped_frame":
            return Gst.FlowReturn.OK

        # Transmit via C++ cryptanetworksink (0x01 = JSON)
        json_payload = json.dumps(result).encode("utf-8")
        packet_bytes = b"\x01" + json_payload
        self._push_to_network(packet_bytes)
        self.stream_a_packets += 1

        return Gst.FlowReturn.OK

    def _on_stream_b_sample(self, appsink: Gst.Element) -> Gst.FlowReturn:
        """
        Stream B: Obfuscates face regions, extracts VAE latents [1, 128],
        encrypts via TenSEAL CKKS, and pushes ciphertext into cryptanetworksink (Type 0x02).
        """
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        pts_ns = buf.pts

        try:
            with MappedBuffer(buf, Gst.MapFlags.READ) as (info, data):
                frame = (
                    np.frombuffer(data, dtype=np.uint8)
                    .reshape(FRAME_HEIGHT, FRAME_WIDTH, 4)
                    .copy()[..., :3]
                )
        except Exception as exc:
            log.warning("Stream B buffer mapping error: %s", exc)
            return Gst.FlowReturn.OK

        # Face obfuscation & VAE latent extraction
        self.ai_engine.privacy_obfuscator.obfuscate(frame)

        # Atomically drain newly extracted face latents
        records = self.ai_engine.latent_store.drain()
        for idx, rec in enumerate(records):
            try:
                # On-Edge Homomorphic Encryption (TenSEAL CKKS)
                ckks_bytes = self.crypto_engine.encrypt_latent(rec.mu)

                # Format envelope with face metadata and raw CKKS ciphertext
                envelope = {
                    "frame_id": rec.frame_id,
                    "face_idx": idx,
                    "box": list(rec.box),
                    "pts_ns": pts_ns,
                    "ckks_bytes_len": len(ckks_bytes),
                }
                envelope_bytes = json.dumps(envelope).encode("utf-8")
                # 4-byte envelope length prefix followed by envelope JSON + CKKS bytes
                env_len_prefix = len(envelope_bytes).to_bytes(4, byteorder="big")
                payload = env_len_prefix + envelope_bytes + ckks_bytes

                # Transmit via C++ cryptanetworksink (0x02 = CKKS)
                packet_bytes = b"\x02" + payload
                self._push_to_network(packet_bytes)
                self.stream_b_packets += 1

            except Exception as exc:
                log.error("Failed to encrypt face latent on local edge: %s", exc)

        return Gst.FlowReturn.OK

    def _push_to_network(self, data_bytes: bytes) -> None:
        """Pushes binary packet into the src_b appsrc linked to cryptanetworksink."""
        if self._tx_appsrc is None:
            return

        gst_buf = Gst.Buffer.new_wrapped(data_bytes)
        with self._tx_lock:
            ret = self._tx_appsrc.emit("push-buffer", gst_buf)
            if ret != Gst.FlowReturn.OK:
                log.debug("src_b push-buffer return code: %s", ret)

    # ── Pipeline Control & Bus Management ─────────────────────────────────────

    def start(self) -> None:
        """Starts GStreamer pipeline on local machine."""
        if self._running:
            return

        log.info(
            "Starting Local Edge Pipeline (Source: %s -> Cloud: %s:%d)...",
            self.source_uri,
            self.cloud_host,
            self.cloud_port,
        )
        self._stopping = False
        self._segment_started = False
        self._eos_event.clear()
        self._pipeline = self._build_pipeline()

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self._loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(target=self._loop.run, name="local-glib-loop", daemon=True)
        self._loop_thread.start()

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to transition Local Edge pipeline to PLAYING state.")

        self._running = True
        log.info("Local Edge Pipeline ACTIVE. Ingesting and processing frames...")

    def stop(self) -> None:
        """Gracefully halts pipeline."""
        if not self._running:
            return

        log.info("Stopping Local Edge Pipeline...")
        self._stopping = True
        if self._tx_appsrc:
            self._tx_appsrc.emit("end-of-stream")

        if self._pipeline:
            self._pipeline.send_event(Gst.Event.new_eos())
            self._eos_event.wait(timeout=3.0)
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None

        if self._loop and self._loop.is_running():
            self._loop.quit()

        self._running = False
        log.info(
            "Local Edge Pipeline Stopped. Metrics: in_frames=%d, stream_a_tx=%d, stream_b_tx=%d",
            self.frames_in,
            self.stream_a_packets,
            self.stream_b_packets,
        )

    def _seek_video_segment(self, initial: bool = False) -> bool:
        """Seek only video ingest; never flush the independent TCP appsrc branch."""
        flags = Gst.SeekFlags.SEGMENT | Gst.SeekFlags.ACCURATE
        if initial:
            flags |= Gst.SeekFlags.FLUSH
        return bool(self._video_source and self._video_source.seek(
            1.0, Gst.Format.TIME, flags,
            Gst.SeekType.SET, 0, Gst.SeekType.NONE, -1,
        ))

    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message) -> bool:
        if (message.type == Gst.MessageType.ASYNC_DONE and self._loop_video
                and not self._segment_started and not self._stopping):
            self._segment_started = self._seek_video_segment(initial=True)
            if not self._segment_started:
                log.error("Local video does not support segment seeking; looping unavailable.")
        elif (message.type == Gst.MessageType.SEGMENT_DONE and self._loop_video
              and not self._stopping):
            if self._seek_video_segment():
                self.loop_count += 1
                log.info("Local video loop %d queued.", self.loop_count)
            else:
                log.error("Failed to queue the next local video segment.")
        elif message.type == Gst.MessageType.EOS:
            log.info("EOS encountered on Local Edge pipeline.")
            self._eos_event.set()
        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            log.error("GStreamer Error from %s: %s (%s)", message.src.get_name(), err, debug)
        return True


# Backward compatibility alias
JetsonEdgePipeline = LocalEdgePipeline


# ── Standalone CLI Runner ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CryptaVision Local Edge AI Node (Hardware-Agnostic)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Input video feed: 'webcam', '0', RTSP URL ('rtsp://...'), HTTP URL ('http://...'), or local file path (.mp4)",
    )
    parser.add_argument(
        "--cloud-host",
        default="127.0.0.1",
        help="Target Cloud TCP Server host",
    )
    parser.add_argument(
        "--cloud-port",
        type=int,
        default=5555,
        help="Target Cloud TCP Server port",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Semantic frame skip interval (1 = process every frame)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Face latent dimension (128 for TenSEAL CKKS)",
    )
    args = parser.parse_args()

    pipeline = LocalEdgePipeline(
        source_uri=args.source,
        cloud_host=args.cloud_host,
        cloud_port=args.cloud_port,
        process_interval=args.interval,
        latent_dim=args.latent_dim,
    )

    def _sig_handler(sig, frame):
        log.info("Caught signal %d; terminating local edge pipeline...", sig)
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    pipeline.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pipeline.stop()


if __name__ == "__main__":
    main()
