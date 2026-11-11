#!/usr/bin/env python3
"""
CryptaVision Phase 6 — NVIDIA Jetson Edge Video Analytics Pipeline
===================================================================
Hardware-accelerated edge runner designed for NVIDIA Jetson Orin Nano,
Xavier NX, and AGX Orin embedded systems.

Edge Architecture & Systems Model:
----------------------------------
1. Hardware-Accelerated Video Ingest:
   - RTSP or CSI camera input.
   - Hardware decoding via Jetson NVDEC (`nvv4l2decoder`).
   - Hardware VIC colorspace conversion (`nvvidconv`) to BGRx.
   - Zero-copy `tee` branching into decoupled memory queues.

2. On-Device AI & Homomorphic Encryption (Zero Raw Video Egress):
   - Stream A: YOLOv8 pose keypoints & CLIP FP16 embeddings.
   - Stream B: Face detection, VAE latent extraction ([1, 128] FP32),
     and on-edge TenSEAL CKKS encryption (scale=2^40).
   - The edge device NEVER transmits raw or obfuscated video frames over the WAN.

3. Low-Level C++ Network Sink Hand-off:
   - Python AI threads serialize payloads and push them into an internal
     `appsrc` linked directly to our custom C++ `cryptanetworksink`.
   - The C++ sink writes directly to raw POSIX TCP sockets with non-blocking I/O,
     completely bypassing Python for the network transmission.
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gst

# Import Phase 1, 2, and 3 Core Engines
from cryptavision_pipeline import MappedBuffer, FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE
from ai_engine import CryptaAIEngine
from crypto_engine import CryptoEngine, TENSEAL_AVAILABLE

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [JetsonEdge]: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("JetsonEdge")


class JetsonEdgePipeline:
    """
    Hardware-accelerated edge pipeline running on NVIDIA Jetson.
    Ingests video, runs local AI inference and CKKS encryption, and pushes
    telemetry through the custom C++ `cryptanetworksink`.
    """

    def __init__(
        self,
        source_uri: str,
        cloud_host: str = "127.0.0.1",
        cloud_port: int = 5555,
        process_interval: int = 5,
        latent_dim: int = 128,
        use_hardware_accel: bool = True,
    ):
        self.source_uri = source_uri
        self.cloud_host = cloud_host
        self.cloud_port = cloud_port
        self.process_interval = process_interval
        self.latent_dim = latent_dim
        self.use_hardware_accel = use_hardware_accel

        self._pipeline: Optional[Gst.Pipeline] = None
        self._loop: Optional[GLib.MainLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._eos_event = threading.Event()
        self._running = False

        # Internal appsrc handle for passing telemetry to C++ cryptanetworksink
        self._tx_appsrc: Optional[Gst.Element] = None
        self._tx_lock = threading.Lock()

        # Telemetry metrics
        self.frames_in = 0
        self.stream_a_packets = 0
        self.stream_b_packets = 0

        # Initialize GStreamer & AI/Crypto engines
        Gst.init(None)
        self._init_engines()

    def _init_engines(self) -> None:
        """Instantiates Jetson-optimized AI engine and TenSEAL CKKS crypto engine."""
        log.info("Initializing Jetson Edge AI Engine (YOLOv8 + CLIP + FaceVAE)...")
        self.ai_engine = CryptaAIEngine(
            process_interval=self.process_interval,
            latent_dim=self.latent_dim,
        )

        log.info("Initializing On-Edge TenSEAL CKKS Homomorphic Encryption Engine...")
        if not TENSEAL_AVAILABLE:
            raise RuntimeError("TenSEAL is required on edge device. Install via pip install tenseal>=0.3.14")
        self.crypto_engine = CryptoEngine(latent_dim=self.latent_dim)
        log.info("CKKS Engine Ready (N=8192, [60,40,40,60], scale=2^40).")

    def _detect_jetson_hardware(self) -> bool:
        """Verifies presence of JetPack hardware decoder elements."""
        nvdec = Gst.ElementFactory.find("nvv4l2decoder")
        nvconv = Gst.ElementFactory.find("nvvidconv")
        return nvdec is not None and nvconv is not None

    def _build_pipeline(self) -> Gst.Pipeline:
        """
        Builds the GStreamer pipeline graph with Jetson NVDEC acceleration
        and raw C++ cryptanetworksink.
        """
        pipeline = Gst.Pipeline.new("jetson-edge-pipeline")

        has_jetson = self._detect_jetson_hardware() if self.use_hardware_accel else False
        if has_jetson:
            log.info("NVIDIA Jetson hardware acceleration detected (nvv4l2decoder + nvvidconv).")
        else:
            log.warning("Jetson NVDEC not found; falling back to software decoders (avdec_h264 + videoconvert).")

        # ── 1. Video Ingest Elements ──────────────────────────────────────────
        is_rtsp = self.source_uri.lower().startswith("rtsp://")
        if is_rtsp:
            src = Gst.ElementFactory.make("rtspsrc", "rtspsrc")
            src.set_property("location", self.source_uri)
            src.set_property("latency", 100)
            src.set_property("protocols", "tcp")
            depay = Gst.ElementFactory.make("rtph264depay", "depay")
        else:
            src = Gst.ElementFactory.make("filesrc", "filesrc")
            src.set_property("location", self.source_uri)
            depay = Gst.ElementFactory.make("qtdemux", "demux")

        parse = Gst.ElementFactory.make("h264parse", "parse")

        # ── 2. Hardware Decoder & Color Conversion ────────────────────────────
        if has_jetson:
            decoder = Gst.ElementFactory.make("nvv4l2decoder", "nv_decoder")
            converter = Gst.ElementFactory.make("nvvidconv", "nv_converter")
            # Output format for Jetson hardware converter
            caps = Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={FRAME_WIDTH},height={FRAME_HEIGHT},framerate={FRAME_RATE}/1"
            )
        else:
            decoder = Gst.ElementFactory.make("avdec_h264", "sw_decoder")
            decoder.set_property("max-threads", 4)
            converter = Gst.ElementFactory.make("videoconvert", "sw_converter")
            caps = Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={FRAME_WIDTH},height={FRAME_HEIGHT},framerate={FRAME_RATE}/1"
            )

        capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        capsfilter.set_property("caps", caps)

        # ── 3. Zero-Copy Fork (Tee) ───────────────────────────────────────────
        tee = Gst.ElementFactory.make("tee", "tee")

        # Branch A: Semantic Analytics (YOLO + CLIP)
        queue_a = Gst.ElementFactory.make("queue", "queue_a")
        queue_a.set_property("leaky", 2)
        queue_a.set_property("max-size-buffers", 3)
        sink_a = Gst.ElementFactory.make("appsink", "appsink_a")
        sink_a.set_property("emit-signals", True)
        sink_a.set_property("sync", False)
        sink_a.set_property("drop", True)
        sink_a.set_property("max-buffers", 1)

        # Branch B: Privacy Face Obfuscation & VAE Latent Extraction
        queue_b = Gst.ElementFactory.make("queue", "queue_b")
        queue_b.set_property("leaky", 2)
        queue_b.set_property("max-size-buffers", 3)
        sink_b = Gst.ElementFactory.make("appsink", "appsink_b")
        sink_b.set_property("emit-signals", True)
        sink_b.set_property("sync", False)
        sink_b.set_property("drop", True)
        sink_b.set_property("max-buffers", 1)

        # ── 4. Network Telemetry Egress Sub-Pipeline ──────────────────────────
        # appsrc receives serialized JSON and CKKS bytes from Python callbacks
        # and feeds directly into the custom C++ cryptanetworksink
        tx_appsrc = Gst.ElementFactory.make("appsrc", "tx_appsrc")
        tx_appsrc.set_property("is-live", True)
        tx_appsrc.set_property("format", Gst.Format.BYTES)
        tx_appsrc.set_property("block", False)
        self._tx_appsrc = tx_appsrc

        net_sink = Gst.ElementFactory.make("cryptanetworksink", "net_sink")
        if net_sink is None:
            # If the C++ plugin is not yet installed into system path, provide helpful diagnostics
            raise RuntimeError(
                "GStreamer element 'cryptanetworksink' not found! "
                "Ensure libcrypta_network_sink.so is compiled and GST_PLUGIN_PATH is set. "
                "Run: cd plugins && mkdir -p build && cd build && cmake .. && make"
            )

        net_sink.set_property("host", self.cloud_host)
        net_sink.set_property("port", self.cloud_port)
        net_sink.set_property("timeout-ms", 1000)
        net_sink.set_property("auto-reconnect", True)
        net_sink.set_property("sync", False)

        # ── 5. Add Elements to Pipeline ───────────────────────────────────────
        elements = [
            src, depay, parse, decoder, converter, capsfilter, tee,
            queue_a, sink_a, queue_b, sink_b, tx_appsrc, net_sink,
        ]
        for el in elements:
            pipeline.add(el)

        # ── 6. Link Graph ─────────────────────────────────────────────────────
        if is_rtsp:
            def _on_rtsp_pad_added(src_el, pad, target):
                sink_pad = target.get_static_pad("sink")
                if not sink_pad.is_linked():
                    pad.link(sink_pad)
            src.connect("pad-added", _on_rtsp_pad_added, depay)
        else:
            src.link(depay)
            def _on_demux_pad_added(demux_el, pad, target):
                sink_pad = target.get_static_pad("sink")
                if not sink_pad.is_linked():
                    pad.link(sink_pad)
            depay.connect("pad-added", _on_demux_pad_added, parse)

        parse.link(decoder)
        decoder.link(converter)
        converter.link(capsfilter)
        capsfilter.link(tee)

        tee.link(queue_a)
        queue_a.link(sink_a)

        tee.link(queue_b)
        queue_b.link(sink_b)

        # Link Network Output
        tx_appsrc.link(net_sink)

        # Wire callbacks
        sink_a.connect("new-sample", self._on_stream_a_sample)
        sink_b.connect("new-sample", self._on_stream_b_sample)

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
                log.error("Failed to encrypt face latent on edge: %s", exc)

        return Gst.FlowReturn.OK

    def _push_to_network(self, data_bytes: bytes) -> None:
        """Pushes binary packet into the tx_appsrc linked to cryptanetworksink."""
        if self._tx_appsrc is None:
            return

        gst_buf = Gst.Buffer.new_wrapped(data_bytes)
        with self._tx_lock:
            ret = self._tx_appsrc.emit("push-buffer", gst_buf)
            if ret != Gst.FlowReturn.OK:
                log.debug("tx_appsrc push-buffer return code: %s", ret)

    # ── Pipeline Control & Bus Management ─────────────────────────────────────

    def start(self) -> None:
        """Starts GStreamer pipeline on Jetson."""
        if self._running:
            return

        log.info("Starting Jetson Edge Pipeline (Egress: %s:%d)...", self.cloud_host, self.cloud_port)
        self._pipeline = self._build_pipeline()

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self._loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(target=self._loop.run, name="jetson-glib-loop", daemon=True)
        self._loop_thread.start()

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to transition Jetson pipeline to PLAYING state.")

        self._running = True
        log.info("Jetson Edge Pipeline ACTIVE. Processing feed...")

    def stop(self) -> None:
        """Gracefully halts pipeline."""
        if not self._running:
            return

        log.info("Stopping Jetson Edge Pipeline...")
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
            "Jetson Pipeline Stopped. Metrics: in_frames=%d, stream_a_tx=%d, stream_b_tx=%d",
            self.frames_in, self.stream_a_packets, self.stream_b_packets,
        )

    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message) -> bool:
        if message.type == Gst.MessageType.EOS:
            log.info("EOS encountered on Jetson pipeline.")
            self._eos_event.set()
        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            log.error("GStreamer Error from %s: %s (%s)", message.src.get_name(), err, debug)
        return True


# ── Standalone CLI Runner ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CryptaVision Jetson Edge AI Node")
    parser.add_argument("--source", default="rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4",
                        help="RTSP URL, video file, or camera path")
    parser.add_argument("--cloud-host", default="127.0.0.1", help="Target Cloud TCP Server host")
    parser.add_argument("--cloud-port", type=int, default=5555, help="Target Cloud TCP Server port")
    parser.add_argument("--interval", type=int, default=5, help="Semantic frame skip interval")
    parser.add_argument("--disable-nvdec", action="store_true", help="Force software decoders (for testing)")
    args = parser.parse_args()

    pipeline = JetsonEdgePipeline(
        source_uri=args.source,
        cloud_host=args.cloud_host,
        cloud_port=args.cloud_port,
        process_interval=args.interval,
        use_hardware_accel=not args.disable_nvdec,
    )

    def _sig_handler(sig, frame):
        log.info("Caught signal %d; terminating edge pipeline...", sig)
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
