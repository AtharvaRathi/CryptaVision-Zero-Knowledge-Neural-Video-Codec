#!/usr/bin/env python3
"""
CryptaVision — Dual-Stream Privacy-Preserving Video Analytics Pipeline
=======================================================================
Architecture:
  Source → Decode → Convert → Tee ─┬─ Queue A → appsink  (Stream A: Semantic)
                                    └─ Queue B → appsink  (Stream B: Visual/Obfuscated)

Stream A feeds raw frames into a Python AI processing loop (feature extraction).
Stream B feeds frames into a modify → re-encode loop via appsrc → avenc_h264.

GStreamer elements used:
  - rtspsrc / filesrc      : ingest
  - rtph264depay / qtdemux : depacketize
  - avdec_h264             : FFmpeg H.264 software decoder
  - videoconvert           : colorspace normalization (→ BGRx / I420)
  - tee                    : non-destructive stream fork
  - queue                  : per-branch decoupled threading
  - appsink                : pull frames into Python
  - appsrc                 : push modified frames back into pipeline
  - avenc_h264             : FFmpeg H.264 software encoder
  - mp4mux / rtsp_sink     : mux / egress (Stream B)

Memory contract:
  - Every Gst.Buffer mapped with GST_MAP_READ must be unmapped in the same scope.
  - appsink sample references are released immediately after data is copied.
  - EOS is propagated explicitly; pipeline is set to NULL before exit to free
    all GStreamer-owned memory.

Author : CryptaVision Core Team
"""

import sys
import time
import signal
import logging
import threading
import traceback
from typing import Optional, Callable

import numpy as np

# ── GStreamer Python bindings ─────────────────────────────────────────────────
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib, GObject

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("CryptaVision")

# ── Constants ─────────────────────────────────────────────────────────────────
# Frame dimensions must match your source; pipeline caps will enforce these.
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
FRAME_RATE   = 30          # fps
QUEUE_MAX_BUFFERS = 5      # per-branch queue depth (tune for latency vs. smoothness)

# appsink pull timeout — never block longer than this waiting for a frame.
APPSINK_TIMEOUT_NS = 500_000_000   # 500 ms in nanoseconds


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: safe buffer map / unmap context manager
# ══════════════════════════════════════════════════════════════════════════════
class MappedBuffer:
    """
    RAII wrapper around Gst.Buffer.map / unmap.

    Usage:
        with MappedBuffer(buf, Gst.MapFlags.READ) as (info, data):
            frame = np.frombuffer(data, dtype=np.uint8)
    """

    def __init__(self, gst_buffer: Gst.Buffer, flags: Gst.MapFlags = Gst.MapFlags.READ):
        self._buf   = gst_buffer
        self._flags = flags
        self._info: Optional[Gst.MapInfo] = None

    def __enter__(self):
        success, self._info = self._buf.map(self._flags)
        if not success:
            raise RuntimeError("Gst.Buffer.map() failed — buffer may be locked or NULL")
        return self._info, self._info.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always unmap, even if an exception was raised inside the block.
        if self._info is not None:
            self._buf.unmap(self._info)
        return False   # do not suppress exceptions


# ══════════════════════════════════════════════════════════════════════════════
#  Stream A processor — pure Python AI stub
# ══════════════════════════════════════════════════════════════════════════════
class SemanticProcessor:
    """
    Receives raw BGR frames from Stream A and extracts feature representations.
    Replace `process_frame` body with real model inference (e.g., CLIP, DINO).
    """

    def __init__(self):
        self.frame_count = 0

    def process_frame(self, frame_np: np.ndarray, pts_ns: int) -> np.ndarray:
        """
        Args:
            frame_np : uint8 ndarray, shape (H, W, C) in BGR colorspace
            pts_ns   : presentation timestamp in nanoseconds

        Returns:
            feature vector (placeholder: mean colour per channel)
        """
        self.frame_count += 1
        features = frame_np.mean(axis=(0, 1))   # (C,) — replace with real model
        if self.frame_count % FRAME_RATE == 0:
            log.info("SemanticProcessor | frame=%d  pts=%.3fs  feat=%s",
                     self.frame_count, pts_ns / 1e9, features.round(2))
        return features


# ══════════════════════════════════════════════════════════════════════════════
#  Stream B processor — visual obfuscation stub
# ══════════════════════════════════════════════════════════════════════════════
class VisualObfuscator:
    """
    Receives raw BGR frames from the Stream-B appsink, applies a privacy filter,
    and pushes the modified frame back via the Stream-B appsrc.

    The modify → push loop runs in its own thread to avoid blocking the
    GStreamer streaming threads.
    """

    def __init__(self):
        self.frame_count = 0

    def obfuscate(self, frame_np: np.ndarray) -> np.ndarray:
        """
        Privacy transform applied to Stream B frames.
        Placeholder: horizontal flip.  Replace with pixelation, silhouette
        extraction, optical-flow masking, etc.
        """
        # np.fliplr is a zero-copy view, so we force a writable copy for safety
        return np.ascontiguousarray(np.fliplr(frame_np))


# ══════════════════════════════════════════════════════════════════════════════
#  CryptaVision Pipeline
# ══════════════════════════════════════════════════════════════════════════════
class CryptaVisionPipeline:
    """
    Dual-stream GStreamer pipeline.

    Constructor parameters
    ----------------------
    source_uri : str
        RTSP URL  (rtsp://...)  or  file path  (/path/to/file.mp4).
    on_semantic_frame : Callable, optional
        External callback invoked for every Stream-A frame.  Signature:
            callback(frame: np.ndarray, pts_ns: int) -> None
    on_visual_frame : Callable, optional
        External callback invoked for every (obfuscated) Stream-B frame.
    """

    def __init__(
        self,
        source_uri: str,
        on_semantic_frame: Optional[Callable] = None,
        on_visual_frame:   Optional[Callable] = None,
    ):
        self.source_uri         = source_uri
        self.on_semantic_frame  = on_semantic_frame
        self.on_visual_frame    = on_visual_frame

        self._pipeline: Optional[Gst.Pipeline] = None
        self._loop:     Optional[GLib.MainLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._eos_event = threading.Event()
        self._running   = False

        self._semantic_processor = SemanticProcessor()
        self._visual_obfuscator  = VisualObfuscator()

        # appsrc handle stored so the obfuscation thread can push buffers
        self._visual_appsrc: Optional[Gst.Element] = None
        self._visual_push_lock = threading.Lock()

        Gst.init(None)

    # ─────────────────────────────────────────────────────────────────────────
    #  Pipeline construction
    # ─────────────────────────────────────────────────────────────────────────
    def _is_rtsp(self) -> bool:
        return self.source_uri.lower().startswith("rtsp://")

    def _build_pipeline(self) -> Gst.Pipeline:
        """
        Assembles the GStreamer element graph programmatically.
        Using code (not parse_launch) gives fine-grained property control
        and makes element handles available for signal connections.
        """

        pipeline = Gst.Pipeline.new("cryptavision")

        # ── 1. Source ─────────────────────────────────────────────────────────
        if self._is_rtsp():
            src = self._make("rtspsrc",  "src")
            src.set_property("location",       self.source_uri)
            src.set_property("latency",        100)          # jitter buffer ms
            src.set_property("protocols",      "tcp")        # force TCP transport
            src.set_property("buffer-mode",    0)            # auto
            # rtspsrc pads are dynamic; link after pad-added signal
            depay = self._make("rtph264depay", "depay")
            parse = self._make("h264parse",    "parse")
        else:
            src   = self._make("filesrc",  "src")
            src.set_property("location", self.source_uri)
            depay = self._make("qtdemux",  "demux")  # handles MP4 container
            parse = self._make("h264parse", "parse")

        # ── 2. Decode (FFmpeg H.264 software decoder) ─────────────────────────
        decoder = self._make("avdec_h264", "decoder")
        decoder.set_property("max-threads", 4)   # use 4 CPU threads for decode

        # ── 3. Colour-space normalisation ─────────────────────────────────────
        # Downstream caps are BGRx so numpy can consume frames directly as BGR.
        # Stream A and B share the same decoded colorspace before the tee.
        convert_pre = self._make("videoconvert", "convert_pre")
        caps_pre    = Gst.Caps.from_string(
            f"video/x-raw,format=BGRx,width={FRAME_WIDTH},"
            f"height={FRAME_HEIGHT},framerate={FRAME_RATE}/1"
        )
        capsfilter_pre = self._make("capsfilter", "capsfilter_pre")
        capsfilter_pre.set_property("caps", caps_pre)

        # ── 4. Tee — non-destructive 1-to-N fork ─────────────────────────────
        #  tee allocates separate downstream queues, each with its own ref.
        #  The original buffer is NOT copied — downstream elements receive
        #  read-only refs until someone requests a writable copy.
        tee = self._make("tee", "tee")
        tee.set_property("allow-not-linked", False)  # error if a branch drops

        # ── 5a. Stream A — Semantic branch ────────────────────────────────────
        queue_a  = self._make("queue",    "queue_a")
        queue_a.set_property("max-size-buffers", QUEUE_MAX_BUFFERS)
        queue_a.set_property("max-size-bytes",   0)
        queue_a.set_property("max-size-time",    0)
        queue_a.set_property("leaky",            2)   # leaky-downstream: drop old

        sink_a = self._make("appsink", "appsink_a")
        sink_a.set_property("emit-signals",  True)
        sink_a.set_property("sync",          False)   # pull as fast as possible
        sink_a.set_property("drop",          True)    # never block the pipeline
        sink_a.set_property("max-buffers",   1)       # keep only latest frame

        # ── 5b. Stream B — Visual / Obfuscation branch ────────────────────────
        queue_b  = self._make("queue",    "queue_b")
        queue_b.set_property("max-size-buffers", QUEUE_MAX_BUFFERS)
        queue_b.set_property("max-size-bytes",   0)
        queue_b.set_property("max-size-time",    0)
        queue_b.set_property("leaky",            2)

        sink_b = self._make("appsink", "appsink_b")
        sink_b.set_property("emit-signals",  True)
        sink_b.set_property("sync",          False)
        sink_b.set_property("drop",          True)
        sink_b.set_property("max-buffers",   1)

        # ── 6. Stream B re-encode sub-pipeline ───────────────────────────────
        # appsrc re-injects obfuscated frames; avenc_h264 re-encodes them.
        appsrc_b = self._make("appsrc",    "appsrc_b")
        appsrc_b.set_property("is-live",       True)
        appsrc_b.set_property("block",         False)
        appsrc_b.set_property("format",        Gst.Format.TIME)
        appsrc_b.set_property("max-bytes",     0)
        appsrc_b.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw,format=BGRx,width={FRAME_WIDTH},"
            f"height={FRAME_HEIGHT},framerate={FRAME_RATE}/1"
        ))
        self._visual_appsrc = appsrc_b

        convert_b = self._make("videoconvert", "convert_b")
        caps_enc  = Gst.Caps.from_string("video/x-raw,format=I420")  # avenc_h264 wants I420
        capsfilter_enc = self._make("capsfilter", "capsfilter_enc")
        capsfilter_enc.set_property("caps", caps_enc)

        encoder_b = self._make("avenc_h264", "encoder_b")
        encoder_b.set_property("bitrate",    2_000_000)     # 2 Mbps
        encoder_b.set_property("pass",       0)             # constant bitrate
        encoder_b.set_property("max-threads", 4)

        # Mux + file sink (for demo; swap for rtspsink / udpsink in production)
        mux_b     = self._make("mp4mux",  "mux_b")
        filesink_b = self._make("filesink", "filesink_b")
        filesink_b.set_property("location", "/tmp/cryptavision_stream_b.mp4")
        filesink_b.set_property("sync",     False)

        # ── Add all elements to the pipeline ─────────────────────────────────
        for el in [
            src, depay, parse, decoder, convert_pre, capsfilter_pre,
            tee,
            queue_a, sink_a,
            queue_b, sink_b,
            appsrc_b, convert_b, capsfilter_enc, encoder_b, mux_b, filesink_b,
        ]:
            pipeline.add(el)

        # ── Static links ──────────────────────────────────────────────────────
        if self._is_rtsp():
            # rtspsrc has dynamic pads — link depay after pad-added fires
            src.connect("pad-added", self._on_rtsp_pad_added, depay)
        else:
            src.link(depay)
            # qtdemux also has dynamic pads (video_%u)
            depay.connect("pad-added", self._on_demux_pad_added, parse)

        parse.link(decoder)
        decoder.link(convert_pre)
        convert_pre.link(capsfilter_pre)
        capsfilter_pre.link(tee)

        # tee → Stream A
        tee.link(queue_a)
        queue_a.link(sink_a)

        # tee → Stream B (appsink for raw capture)
        tee.link(queue_b)
        queue_b.link(sink_b)

        # Re-encode sub-pipeline: appsrc_b → convert_b → encoder_b → mux_b → filesink_b
        appsrc_b.link(convert_b)
        convert_b.link(capsfilter_enc)
        capsfilter_enc.link(encoder_b)
        encoder_b.link(mux_b)
        mux_b.link(filesink_b)

        # ── Signal connections ────────────────────────────────────────────────
        sink_a.connect("new-sample", self._on_new_sample_a)
        sink_b.connect("new-sample", self._on_new_sample_b)

        return pipeline

    # ─────────────────────────────────────────────────────────────────────────
    #  Dynamic pad handlers
    # ─────────────────────────────────────────────────────────────────────────
    def _on_rtsp_pad_added(self, src_el: Gst.Element, new_pad: Gst.Pad,
                           depay: Gst.Element) -> None:
        """Links rtspsrc's dynamic video pad → rtph264depay."""
        sink_pad = depay.get_static_pad("sink")
        if sink_pad.is_linked():
            return   # already linked; ignore extra pads (audio, metadata)
        caps = new_pad.get_current_caps() or new_pad.query_caps(None)
        struct = caps.get_structure(0)
        if "application/x-rtp" in struct.get_name():
            ret = new_pad.link(sink_pad)
            if ret != Gst.PadLinkReturn.OK:
                log.error("rtspsrc pad link failed: %s", ret)
            else:
                log.info("RTSP pad linked: %s", new_pad.get_name())

    def _on_demux_pad_added(self, demux: Gst.Element, new_pad: Gst.Pad,
                             parse: Gst.Element) -> None:
        """Links qtdemux's dynamic video_%u pad → h264parse."""
        sink_pad = parse.get_static_pad("sink")
        if sink_pad.is_linked():
            return
        caps   = new_pad.get_current_caps() or new_pad.query_caps(None)
        struct = caps.get_structure(0)
        if "video" in struct.get_name():
            ret = new_pad.link(sink_pad)
            if ret != Gst.PadLinkReturn.OK:
                log.error("qtdemux pad link failed: %s", ret)
            else:
                log.info("Demux video pad linked: %s", new_pad.get_name())

    # ─────────────────────────────────────────────────────────────────────────
    #  appsink callbacks  (run in GStreamer streaming thread)
    # ─────────────────────────────────────────────────────────────────────────
    def _on_new_sample_a(self, appsink: Gst.Element) -> Gst.FlowReturn:
        """
        Stream A — called every time the appsink has a new decoded frame.

        Strict memory contract:
          1. Pull sample (increments ref-count on internal buffer).
          2. Extract Gst.Buffer from sample.
          3. Map buffer READ-ONLY inside MappedBuffer context manager.
          4. Copy raw bytes into a numpy array (zero-extra-allocation: uses
             np.frombuffer which shares memory, then .copy() to own the data).
          5. Context manager unmaps buffer on __exit__.
          6. sample is released (ref-count decremented) at function exit.
        """
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf      = sample.get_buffer()
        pts_ns   = buf.pts           # presentation timestamp in nanoseconds

        try:
            with MappedBuffer(buf, Gst.MapFlags.READ) as (info, data):
                # BGRx: 4 bytes per pixel; slice off the unused X channel in numpy
                frame = (
                    np.frombuffer(data, dtype=np.uint8)
                      .reshape(FRAME_HEIGHT, FRAME_WIDTH, 4)
                      .copy()          # own the data before unmap
                    [..., :3]          # drop X → BGR  (H, W, 3)
                )
        except RuntimeError as exc:
            log.warning("Stream A buffer map error: %s", exc)
            return Gst.FlowReturn.OK   # keep pipeline running

        # ── Hand off to AI layer ───────────────────────────────────────────
        self._semantic_processor.process_frame(frame, pts_ns)
        if self.on_semantic_frame:
            try:
                self.on_semantic_frame(frame, pts_ns)
            except Exception:
                log.exception("on_semantic_frame callback raised")

        return Gst.FlowReturn.OK

    def _on_new_sample_b(self, appsink: Gst.Element) -> Gst.FlowReturn:
        """
        Stream B — pull raw frame, obfuscate it, push into appsrc_b.

        The obfuscation runs synchronously here; for heavy transforms move it
        to a thread pool and push to appsrc_b from that thread.
        """
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf    = sample.get_buffer()
        pts_ns = buf.pts
        dur_ns = buf.duration

        try:
            with MappedBuffer(buf, Gst.MapFlags.READ) as (info, data):
                raw = (
                    np.frombuffer(data, dtype=np.uint8)
                      .reshape(FRAME_HEIGHT, FRAME_WIDTH, 4)
                      .copy()
                )
        except RuntimeError as exc:
            log.warning("Stream B buffer map error: %s", exc)
            return Gst.FlowReturn.OK

        # ── Apply privacy transform ────────────────────────────────────────
        obfuscated = self._visual_obfuscator.obfuscate(raw)   # (H, W, 4) BGRx

        if self.on_visual_frame:
            try:
                self.on_visual_frame(obfuscated[..., :3], pts_ns)
            except Exception:
                log.exception("on_visual_frame callback raised")

        # ── Push obfuscated frame into re-encode pipeline ──────────────────
        self._push_to_appsrc(obfuscated, pts_ns, dur_ns)

        return Gst.FlowReturn.OK

    def _push_to_appsrc(self, frame_bgrx: np.ndarray,
                        pts_ns: int, dur_ns: int) -> None:
        """
        Wraps a numpy array in a Gst.Buffer and pushes it to appsrc_b.

        Memory contract:
          - Gst.Buffer.new_wrapped() copies the bytes once from numpy into
            GStreamer-managed memory.  The numpy array can be freed immediately
            after this call.
          - pts / duration are preserved from the original buffer so the
            downstream muxer can reconstruct correct timestamps.
        """
        if self._visual_appsrc is None:
            return

        raw_bytes = frame_bgrx.tobytes()           # single copy into bytes object
        new_buf   = Gst.Buffer.new_wrapped(raw_bytes)
        new_buf.pts      = pts_ns
        new_buf.duration = dur_ns

        with self._visual_push_lock:
            ret = self._visual_appsrc.emit("push-buffer", new_buf)

        if ret != Gst.FlowReturn.OK:
            log.warning("appsrc push-buffer returned: %s", ret)

    # ─────────────────────────────────────────────────────────────────────────
    #  Bus message handler
    # ─────────────────────────────────────────────────────────────────────────
    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message) -> bool:
        """
        Handles GStreamer bus messages posted by pipeline elements.
        Returns True to keep watching; False to remove the watch.
        """
        mtype = message.type

        if mtype == Gst.MessageType.EOS:
            log.info("EOS received — stopping pipeline.")
            self._eos_event.set()
            self._loop.quit()

        elif mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            log.error("GStreamer ERROR from %s: %s\n  debug: %s",
                      message.src.get_name(), err.message, debug)
            self._loop.quit()

        elif mtype == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            log.warning("GStreamer WARNING from %s: %s\n  debug: %s",
                        message.src.get_name(), warn.message, debug)

        elif mtype == Gst.MessageType.STATE_CHANGED:
            # Only log state changes of the top-level pipeline
            if message.src == self._pipeline:
                old, new, pending = message.parse_state_changed()
                log.debug("Pipeline state: %s → %s (pending: %s)",
                          old.value_nick, new.value_nick, pending.value_nick)

        elif mtype == Gst.MessageType.STREAM_START:
            log.info("Stream started.")

        elif mtype == Gst.MessageType.QOS:
            # Quality-of-service drops — log but don't abort
            _live, _running, _stream, _ts, qos_stats = message.parse_qos()
            log.debug("QOS drop from %s — jitter=%dns",
                      message.src.get_name(), qos_stats)

        return True   # keep watching

    # ─────────────────────────────────────────────────────────────────────────
    #  Lifecycle
    # ─────────────────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Build pipeline, wire bus, and transition to PLAYING state."""
        if self._running:
            raise RuntimeError("Pipeline already running")

        log.info("Building pipeline for source: %s", self.source_uri)
        self._pipeline = self._build_pipeline()

        # Attach bus watch on the GLib main loop
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        # Run GLib main loop on a dedicated background thread so start()
        # returns immediately and the caller can register callbacks.
        self._loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(
            target=self._loop.run,
            name="glib-main-loop",
            daemon=True,
        )
        self._loop_thread.start()

        # Transition pipeline to PLAYING
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to set pipeline to PLAYING")

        self._running = True
        log.info("Pipeline is PLAYING.")

    def stop(self) -> None:
        """Gracefully tear down the pipeline and free GStreamer resources."""
        if not self._running:
            return

        log.info("Sending EOS into pipeline…")

        # Signal appsrc_b that there are no more buffers
        if self._visual_appsrc is not None:
            self._visual_appsrc.emit("end-of-stream")

        # Send EOS event into the main source pad
        self._pipeline.send_event(Gst.Event.new_eos())

        # Wait up to 5 s for EOS to propagate through all sinks
        if not self._eos_event.wait(timeout=5.0):
            log.warning("EOS did not propagate within 5 s — forcing NULL state.")

        # GStreamer state transitions must follow the order:
        # PLAYING → PAUSED → READY → NULL
        # NULL releases all hardware / file / network resources.
        self._pipeline.set_state(Gst.State.NULL)
        self._running = False

        if self._loop and self._loop.is_running():
            self._loop.quit()

        log.info("Pipeline stopped and resources released.")

    def wait_for_eos(self, timeout: Optional[float] = None) -> bool:
        """Block the calling thread until EOS or timeout (seconds)."""
        return self._eos_event.wait(timeout=timeout)

    # ─────────────────────────────────────────────────────────────────────────
    #  Utility
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _make(factory: str, name: str) -> Gst.Element:
        """
        Creates a GStreamer element, raising a descriptive error if the factory
        is not found (e.g., missing plugin package).
        """
        el = Gst.ElementFactory.make(factory, name)
        if el is None:
            raise RuntimeError(
                f"Could not create element '{factory}' ('{name}'). "
                f"Is the GStreamer plugin installed?  "
                f"Try: gst-inspect-1.0 {factory}"
            )
        return el


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="CryptaVision — Dual-Stream Privacy-Preserving Analytics"
    )
    parser.add_argument(
        "--source",
        default="rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4",
        help="RTSP URL or local file path (.mp4/.mkv)",
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="How many seconds to run before auto-stopping (0 = run until EOS)",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Optionally raise GStreamer log level:
        # Gst.debug_set_default_threshold(Gst.DebugLevel.INFO)

    # ── Optional external callbacks ────────────────────────────────────────
    def on_semantic(frame: np.ndarray, pts_ns: int) -> None:
        """Plug your ML inference here."""
        pass

    def on_visual(frame: np.ndarray, pts_ns: int) -> None:
        """Plug your display / streaming egress here."""
        pass

    cv = CryptaVisionPipeline(
        source_uri         = args.source,
        on_semantic_frame  = on_semantic,
        on_visual_frame    = on_visual,
    )

    # ── Graceful Ctrl-C handler ────────────────────────────────────────────
    def _sigint_handler(signum, frame):
        log.info("Caught SIGINT — stopping…")
        cv.stop()

    signal.signal(signal.SIGINT,  _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    cv.start()

    try:
        if args.duration > 0:
            log.info("Running for %.1f seconds…", args.duration)
            eos = cv.wait_for_eos(timeout=args.duration)
            if not eos:
                log.info("Duration elapsed — stopping pipeline.")
        else:
            log.info("Running until EOS (Ctrl-C to stop).")
            cv.wait_for_eos()
    finally:
        cv.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
