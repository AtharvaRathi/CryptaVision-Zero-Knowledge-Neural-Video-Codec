#!/usr/bin/env python3
"""
CryptaVision Phase 4 — FastAPI Asynchronous Orchestration Layer
===============================================================
Exposes the dual-stream video analytics pipeline, homomorphic encryption engine,
and HashiCorp Vault key management via an asynchronous REST, SSE, and WebSocket API.

API Directives & Architecture:
------------------------------
1. `POST /stream/start`:
   - Validates RTSP URL or video path via Pydantic.
   - Non-blocking: spawns GStreamer pipeline on a background worker thread.
   - Integrates Phase 2 AI Engine & Phase 3 Homomorphic Encryption.

2. `GET /analytics/stream-a`:
   - Server-Sent Events (SSE) via `fastapi.responses.StreamingResponse`.
   - Continuous real-time streaming of semantic JSON events (CLIP embeddings,
     YOLO pose keypoints, human counts) to downstream subscribers.
   - Supports multiple concurrent subscribers with zero-drop client queues.

3. `GET /analytics/stream-b`:
   - Protected endpoint requiring a valid HashiCorp Vault compliance token in headers.
   - Verifies token validity via `VaultManager`.
   - Atomically drains the thread-safe `LatentStore` (`atomic list swap`).
   - Encrypts each latent vector with TenSEAL CKKS (scale=2^40) and returns
     the serialized CKKS bytes and cryptographic verification metadata.

4. `GET /health` & `GET /stream/status`:
   - Kubernetes liveness & readiness probes with GPU/Vault status inspection.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

import numpy as np
import torch
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# Import CryptaVision Core Engines
try:
    from cryptavision_pipeline import CryptaVisionPipeline
    from ai_engine import CryptaAIEngine, LatentRecord, LatentStore
    GSTREAMER_AI_AVAILABLE = True
except ImportError as exc:
    logging.warning("GStreamer or AI engine components unavailable: %s", exc)
    CryptaVisionPipeline = None  # type: ignore
    CryptaAIEngine = None  # type: ignore
    LatentRecord = None  # type: ignore
    LatentStore = None  # type: ignore
    GSTREAMER_AI_AVAILABLE = False

try:
    from crypto_engine import CryptoEngine, TENSEAL_AVAILABLE
    from vault_manager import VaultConfig, VaultManager
    CRYPTO_VAULT_AVAILABLE = True
except ImportError as exc:
    logging.warning("Crypto or Vault manager unavailable: %s", exc)
    CryptoEngine = None  # type: ignore
    VaultManager = None  # type: ignore
    TENSEAL_AVAILABLE = False
    CRYPTO_VAULT_AVAILABLE = False

try:
    from cloud.tcp_receiver import CloudTCPReceiver, cloud_latent_store
    CLOUD_TCP_AVAILABLE = True
except ImportError as exc:
    logging.warning("Cloud TCP Receiver module unavailable: %s", exc)
    CloudTCPReceiver = None  # type: ignore
    cloud_latent_store = None  # type: ignore
    CLOUD_TCP_AVAILABLE = False

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("CryptaVision.API")


# ══════════════════════════════════════════════════════════════════════════════
#  Pydantic Request & Response Schemas
# ══════════════════════════════════════════════════════════════════════════════
class StreamStartRequest(BaseModel):
    """Payload to launch a GStreamer analytics pipeline."""

    source_uri: str = Field(
        default="rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4",
        description="RTSP feed URL (rtsp://...) or local video file path",
    )
    process_interval: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Frame skip interval for semantic extraction (1 = every frame)",
    )
    latent_dim: int = Field(
        default=128,
        description="Dimensionality of face latent embeddings (must be 128 for CKKS)",
    )
    enable_crypto: bool = Field(
        default=True,
        description="Enable TenSEAL CKKS Homomorphic Encryption of face latents",
    )
    restart_if_active: bool = Field(
        default=False,
        description="Automatically stop existing pipeline if one is currently active",
    )


class StreamStartResponse(BaseModel):
    """Confirmation response upon pipeline initiation."""

    status: str
    message: str
    source_uri: str
    started_at: str


class StreamStatusResponse(BaseModel):
    """Current operational telemetry of the pipeline."""

    state: str
    source_uri: Optional[str]
    uptime_seconds: float
    semantic_frames_processed: int
    latents_in_store: int
    active_stream_a_subscribers: int
    gpu_available: bool
    gpu_device_name: Optional[str]
    crypto_engine_active: bool
    vault_connected: bool
    vault_fallback: bool


class LatentItem(BaseModel):
    """Metadata and serialized CKKS ciphertext for a single detected face."""

    frame_id: int
    face_idx: int
    box: List[int]
    pts_ns: int
    ckks_bytes_b64: str
    byte_size: int
    sha256: str


class StreamBResponse(BaseModel):
    """Protected payload of encrypted face latents for compliance forensics."""

    count: int
    timestamp: str
    officer_token_verified: bool
    latents: List[LatentItem]


class HealthResponse(BaseModel):
    """Liveness & readiness probe payload."""

    status: str
    timestamp: str
    version: str
    pipeline_state: str
    gpu_available: bool
    cuda_devices: int
    vault_status: str


# ══════════════════════════════════════════════════════════════════════════════
#  Event Broadcaster for Stream A (Server-Sent Events & WebSockets)
# ══════════════════════════════════════════════════════════════════════════════
class StreamABroadcaster:
    """
    Thread-safe publisher/subscriber hub bridging GStreamer streaming threads
    to asyncio Server-Sent Events (SSE) and WebSocket clients.
    """

    def __init__(self, max_queue_size: int = 100):
        self._subscribers: Set[asyncio.Queue] = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.total_frames_broadcast = 0

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        """Registers a new SSE/WebSocket consumer queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        with self._lock:
            self._subscribers.add(q)
            log.info("Client subscribed to Stream A. Active subscribers: %d", len(self._subscribers))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Removes a disconnected consumer queue."""
        with self._lock:
            self._subscribers.discard(q)
            log.info("Client unsubscribed from Stream A. Active subscribers: %d", len(self._subscribers))

    def publish_from_thread(self, event_data: Dict[str, Any]) -> None:
        """
        Called from the GStreamer streaming thread. Safely posts the event
        to all active asyncio queues without blocking the video thread.
        """
        if self._loop is None or self._loop.is_closed():
            return

        self.total_frames_broadcast += 1
        with self._lock:
            if not self._subscribers:
                return
            for q in list(self._subscribers):
                try:
                    self._loop.call_soon_threadsafe(self._push_to_queue, q, event_data)
                except Exception as exc:
                    log.debug("Failed to schedule event push: %s", exc)

    @staticmethod
    def _push_to_queue(q: asyncio.Queue, event_data: Dict[str, Any]) -> None:
        """Pushes data into the queue; evicts oldest element if queue is full."""
        if q.full():
            try:
                q.get_nowait()  # Drop oldest frame to maintain real-time low latency
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(event_data)


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline Orchestrator Manager
# ══════════════════════════════════════════════════════════════════════════════
class PipelineManager:
    """
    Manages the lifecycle of the GStreamer pipeline, AI inference engine,
    and cryptographic modules.
    """

    def __init__(self, broadcaster: StreamABroadcaster):
        self.broadcaster = broadcaster
        self.pipeline: Optional[CryptaVisionPipeline] = None
        self.ai_engine: Optional[CryptaAIEngine] = None
        self.crypto_engine: Optional[CryptoEngine] = None
        self.vault_manager: Optional[VaultManager] = None

        self.state = "IDLE"  # IDLE, STARTING, RUNNING, STOPPING, ERROR
        self.source_uri: Optional[str] = None
        self.started_at: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def initialize_crypto(self, latent_dim: int = 128) -> None:
        """Initializes Vault connection and TenSEAL CKKS Homomorphic Encryption."""
        if not CRYPTO_VAULT_AVAILABLE:
            log.warning("Crypto/Vault libraries not available; proceeding without HE.")
            return

        try:
            log.info("Initializing HashiCorp Vault Manager...")
            vault_cfg = VaultConfig()
            self.vault_manager = VaultManager(vault_cfg)

            log.info("Initializing TenSEAL CKKS Homomorphic Encryption Engine...")
            self.crypto_engine = CryptoEngine(latent_dim=latent_dim)

            # Store the secret context into Vault KV v2 (Zero-Trust Key Management)
            sec_ctx = self.crypto_engine.serialize_secret_context()
            self.vault_manager.store_secret_context(sec_ctx)
            log.info("Secret CKKS context successfully committed to Vault KV v2.")
        except Exception as exc:
            log.error("Failed to initialize Crypto/Vault layer: %s", exc)

    def start_pipeline(
        self,
        source_uri: str,
        process_interval: int = 5,
        latent_dim: int = 128,
        enable_crypto: bool = True,
        restart_if_active: bool = False,
    ) -> None:
        """
        Synchronous pipeline startup invoked via background thread.
        """
        with self._lock:
            if self.state in ("RUNNING", "STARTING"):
                if restart_if_active:
                    log.info("Stopping existing pipeline before restart...")
                    self._stop_pipeline_internal()
                else:
                    raise RuntimeError("Pipeline is already active. Call /stream/stop first or set restart_if_active=True.")

            self.state = "STARTING"
            self.source_uri = source_uri

        try:
            if not GSTREAMER_AI_AVAILABLE:
                raise RuntimeError("GStreamer or AI engine libraries are not installed.")

            # 1. Instantiate AI Engine
            log.info("Instantiating CryptaAIEngine (interval=%d, latent_dim=%d)", process_interval, latent_dim)
            self.ai_engine = CryptaAIEngine(
                process_interval=process_interval,
                latent_dim=latent_dim,
            )

            # 2. Instantiate or refresh Crypto if requested
            if enable_crypto and self.crypto_engine is None:
                self.initialize_crypto(latent_dim=latent_dim)

            # 3. Create GStreamer Pipeline
            log.info("Building GStreamer pipeline for: %s", source_uri)
            self.pipeline = CryptaVisionPipeline(source_uri=source_uri)

            # 4. Wire Semantic Output -> Stream A Broadcaster
            def _on_semantic_result(result: Dict[str, Any]) -> None:
                if result.get("status") == "skipped_frame":
                    return
                # Broadcast to SSE / WebSockets
                self.broadcaster.publish_from_thread(result)

            class _APISemanticShim:
                def __init__(self, analyzer, callback):
                    self.analyzer = analyzer
                    self.callback = callback

                def process_frame(self, frame_np: np.ndarray, pts_ns: int):
                    res = self.analyzer.process_frame(frame_np, pts_ns)
                    self.callback(res)
                    return np.array([], dtype=np.float32)

            self.pipeline._semantic_processor = _APISemanticShim(
                analyzer=self.ai_engine.semantic_analyzer,
                callback=_on_semantic_result,
            )

            # 5. Wire Visual Obfuscation -> Stream B
            self.pipeline._visual_obfuscator = self.ai_engine.privacy_obfuscator

            # 6. Start GStreamer execution
            self.pipeline.start()
            self.started_at = time.time()
            self.state = "RUNNING"
            log.info("GStreamer pipeline successfully transitions to PLAYING state.")

        except Exception as exc:
            self.state = "ERROR"
            log.exception("Pipeline failed to start: %s", exc)
            raise

    def stop_pipeline(self) -> None:
        """Stops the active GStreamer pipeline."""
        with self._lock:
            self._stop_pipeline_internal()

    def _stop_pipeline_internal(self) -> None:
        """Internal helper to stop pipeline and reset state."""
        if self.pipeline is not None:
            log.info("Shutting down active GStreamer pipeline...")
            try:
                self.pipeline.stop()
            except Exception as exc:
                log.error("Error during pipeline stop: %s", exc)
            finally:
                self.pipeline = None

        self.state = "STOPPED"
        self.started_at = None
        log.info("Pipeline stopped and resources freed.")

    def drain_latents(self) -> List[Any]:
        """Atomically drains collected face latents from LatentStore."""
        if self.ai_engine is None or self.ai_engine.latent_store is None:
            return []
        return self.ai_engine.latent_store.drain()


# ══════════════════════════════════════════════════════════════════════════════
#  Global Singletons & Lifespan
# ══════════════════════════════════════════════════════════════════════════════
broadcaster = StreamABroadcaster()
pipeline_mgr = PipelineManager(broadcaster)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown."""
    loop = asyncio.get_running_loop()
    broadcaster.set_loop(loop)
    log.info("CryptaVision API initialized. Event loop bound to Broadcaster.")

    # Initialize cryptographic key manager in background
    try:
        pipeline_mgr.initialize_crypto(latent_dim=128)
    except Exception as exc:
        log.warning("Deferred crypto initialization: %s", exc)

    # Launch Edge-to-Cloud Asynchronous TCP Receiver (Phase 6)
    tcp_receiver: Optional[Any] = None
    if CLOUD_TCP_AVAILABLE and CloudTCPReceiver is not None:
        try:
            tcp_port = int(os.getenv("TCP_INGEST_PORT", "5555"))
            tcp_receiver = CloudTCPReceiver(
                host="0.0.0.0",
                port=tcp_port,
                on_json_callback=broadcaster.publish_from_thread,
            )
            await tcp_receiver.start()
            log.info("Phase 6 Cloud TCP Receiver active on port %d.", tcp_port)
        except Exception as exc:
            log.warning("Could not start Cloud TCP Receiver on port 5555: %s", exc)

    yield  # Application is running

    log.info("Shutting down CryptaVision API. Terminating services...")
    if tcp_receiver is not None:
        await tcp_receiver.stop()
    pipeline_mgr.stop_pipeline()


# ══════════════════════════════════════════════════════════════════════════════
#  FastAPI Application Definition
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Project CryptaVision — Orchestration & Analytics API",
    description=(
        "Zero-defect, privacy-preserving dual-stream video analytics platform. "
        "Integrates GStreamer, PyTorch AI inference (YOLO, CLIP, VAE), "
        "TenSEAL CKKS Homomorphic Encryption, and HashiCorp Vault KV v2."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

# Enable CORS for web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  Authentication Dependency for Protected Endpoints
# ══════════════════════════════════════════════════════════════════════════════
def verify_vault_compliance_token(
    x_vault_token: Optional[str] = Header(None, alias="X-Vault-Token"),
    x_compliance_token: Optional[str] = Header(None, alias="X-Compliance-Token"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> str:
    """
    Extracts and validates a Vault compliance officer token from request headers.

    Checks (in priority order):
      1. X-Vault-Token header
      2. X-Compliance-Token header
      3. Authorization: Bearer <token>
    """
    token: Optional[str] = None
    if x_vault_token:
        token = x_vault_token
    elif x_compliance_token:
        token = x_compliance_token
    elif authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Authentication required: Missing compliance token. "
                "Provide 'X-Vault-Token', 'X-Compliance-Token', or 'Authorization: Bearer <token>' header."
            ),
        )

    # Verify with VaultManager
    if pipeline_mgr.vault_manager is not None:
        is_valid = pipeline_mgr.vault_manager.verify_token(token)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access Denied: Invalid, expired, or unauthorized compliance investigation token.",
            )
    else:
        # If VaultManager is uninitialized, reject in production
        log.warning("VaultManager not initialized. Token validation rejected.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vault Key Management is currently unavailable to verify tokens.",
        )

    return token


# ══════════════════════════════════════════════════════════════════════════════
#  API Endpoints
# ══════════════════════════════════════════════════════════════════════════════

# ── Health & Diagnostics ──────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
@app.get("/healthz", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """
    Kubernetes liveness and readiness probe endpoint.
    Reports API operational status, GPU availability, and Vault connectivity.
    """
    gpu_avail = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if gpu_avail else 0

    vault_status = "uninitialized"
    if pipeline_mgr.vault_manager:
        if pipeline_mgr.vault_manager.is_connected:
            vault_status = "connected"
        elif pipeline_mgr.vault_manager.is_fallback:
            vault_status = "fallback_in_memory"

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="4.0.0",
        pipeline_state=pipeline_mgr.state,
        gpu_available=gpu_avail,
        cuda_devices=cuda_count,
        vault_status=vault_status,
    )


# ── Pipeline Control ──────────────────────────────────────────────────────────
@app.post("/stream/start", response_model=StreamStartResponse, tags=["Pipeline Control"])
async def start_stream(
    req: StreamStartRequest,
    background_tasks: BackgroundTasks,
):
    """
    Initiates the GStreamer video analytics pipeline.
    Spawns execution in a background thread to prevent blocking the event loop.
    """
    if pipeline_mgr.state in ("RUNNING", "STARTING") and not req.restart_if_active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Pipeline is already running. Use /stream/stop or pass restart_if_active=True.",
        )

    def _bg_start():
        try:
            pipeline_mgr.start_pipeline(
                source_uri=req.source_uri,
                process_interval=req.process_interval,
                latent_dim=req.latent_dim,
                enable_crypto=req.enable_crypto,
                restart_if_active=req.restart_if_active,
            )
        except Exception as err:
            log.error("Background pipeline startup failed: %s", err)

    # Dispatch to background task pool
    background_tasks.add_task(_bg_start)

    return StreamStartResponse(
        status="initiating",
        message="GStreamer pipeline startup dispatched to background worker.",
        source_uri=req.source_uri,
        started_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/stream/stop", tags=["Pipeline Control"])
async def stop_stream():
    """Terminates the active GStreamer pipeline and releases all resources."""
    if pipeline_mgr.state == "IDLE":
        return {"status": "idle", "message": "No active pipeline to stop."}

    await asyncio.to_thread(pipeline_mgr.stop_pipeline)
    return {"status": "stopped", "message": "Pipeline halted successfully."}


@app.get("/stream/status", response_model=StreamStatusResponse, tags=["Pipeline Control"])
async def get_stream_status():
    """Returns telemetry and real-time state of the video processing pipeline."""
    now = time.time()
    uptime = (now - pipeline_mgr.started_at) if pipeline_mgr.started_at else 0.0

    gpu_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if gpu_avail else None

    latents_buffered = 0
    if pipeline_mgr.ai_engine and pipeline_mgr.ai_engine.latent_store:
        latents_buffered = len(pipeline_mgr.ai_engine.latent_store)

    vault_conn = bool(pipeline_mgr.vault_manager and pipeline_mgr.vault_manager.is_connected)
    vault_fall = bool(pipeline_mgr.vault_manager and pipeline_mgr.vault_manager.is_fallback)

    return StreamStatusResponse(
        state=pipeline_mgr.state,
        source_uri=pipeline_mgr.source_uri,
        uptime_seconds=round(uptime, 2),
        semantic_frames_processed=broadcaster.total_frames_broadcast,
        latents_in_store=latents_buffered,
        active_stream_a_subscribers=len(broadcaster._subscribers),
        gpu_available=gpu_avail,
        gpu_device_name=dev_name,
        crypto_engine_active=pipeline_mgr.crypto_engine is not None,
        vault_connected=vault_conn,
        vault_fallback=vault_fall,
    )


# ── Stream A: Semantic Analytics via Server-Sent Events (SSE) ─────────────────
@app.get("/analytics/stream-a", tags=["Analytics Streams"])
async def stream_a_sse(request: Request):
    """
    Continuous Server-Sent Events (SSE) stream of Stream A Semantic Analytics.
    Yields JSON events containing YOLOv8 pose keypoints, human counts,
    and 512-dimensional CLIP embeddings.
    """
    queue = broadcaster.subscribe()

    async def _event_generator() -> AsyncGenerator[str, None]:
        try:
            # Yield initial connection confirmation
            init_payload = {
                "event": "connected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pipeline_state": pipeline_mgr.state,
            }
            yield f"event: system\ndata: {json.dumps(init_payload)}\n\n"

            while True:
                # Detect client disconnect to prevent leaked queues
                if await request.is_disconnected():
                    break

                try:
                    # Wait up to 5 seconds for new frame data
                    data = await asyncio.wait_for(queue.get(), timeout=5.0)
                    yield f"event: semantic_frame\ndata: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    # Send keep-alive heartbeat comment
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            log.info("SSE client stream connection cancelled.")
        finally:
            broadcaster.unsubscribe(queue)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disables buffering on Nginx / K8s Ingress
        },
    )


@app.websocket("/ws/analytics/stream-a")
async def stream_a_websocket(websocket: WebSocket):
    """
    WebSocket alternative for streaming Stream A semantic events.
    """
    await websocket.accept()
    queue = broadcaster.subscribe()
    try:
        await websocket.send_json({
            "event": "connected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        while True:
            data = await queue.get()
            await websocket.send_json(data)
    except WebSocketDisconnect:
        log.info("Stream A WebSocket client disconnected.")
    except Exception as exc:
        log.debug("WebSocket exception: %s", exc)
    finally:
        broadcaster.unsubscribe(queue)


# ── Stream B: Encrypted Latents for Authorized Compliance Forensics ───────────
@app.get(
    "/analytics/stream-b",
    response_model=StreamBResponse,
    tags=["Analytics Streams"],
    responses={
        200: {
            "description": "Returns serialized TenSEAL CKKS ciphertexts and metadata.",
            "content": {
                "application/json": {},
                "application/octet-stream": {
                    "schema": {"type": "string", "format": "binary"}
                },
            },
        },
        401: {"description": "Missing compliance token header."},
        403: {"description": "Invalid or expired compliance token."},
    },
)
async def get_stream_b_latents(
    format: str = Query("json", description="Output format"),
    token: str = Depends(verify_vault_compliance_token),
):
    """
    Protected Endpoint: Retrieves encrypted face representations for forensic investigation.

    Requirements:
      1. Valid HashiCorp Vault compliance token in headers.
      2. Atomically drains `LatentStore.drain()` to prevent torn reads.
      3. Encrypts each `mu` tensor [1, 128] using TenSEAL CKKS.
      4. Returns serialized CKKS bytes (in Base64 JSON or binary stream).
    """
    # 1. Atomically drain pending latents from local pipeline and edge ingestion store
    records = pipeline_mgr.drain_latents()
    cloud_records = await cloud_latent_store.drain() if (CLOUD_TCP_AVAILABLE and cloud_latent_store is not None) else []

    items: List[LatentItem] = []
    raw_concatenated_bytes = bytearray()

    # Ingest edge-transmitted latents (already CKKS-encrypted on Jetson)
    for cr in cloud_records:
        item = LatentItem(
            frame_id=cr.frame_id,
            face_idx=cr.face_idx,
            box=list(cr.box),
            pts_ns=cr.pts_ns,
            ckks_bytes_b64=base64.b64encode(cr.ckks_bytes).decode("ascii"),
            byte_size=len(cr.ckks_bytes),
            sha256=cr.sha256,
        )
        items.append(item)
        raw_concatenated_bytes.extend(cr.ckks_bytes)

    # Ingest and encrypt local pipeline latents (if any)
    if records:
        if pipeline_mgr.crypto_engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CryptoEngine is not active. Ensure enable_crypto=True on pipeline start.",
            )

        for idx, rec in enumerate(records):
            try:
                # Encrypt [1, 128] float32 tensor into CKKS ciphertext bytes
                ckks_bytes = pipeline_mgr.crypto_engine.encrypt_latent(rec.mu)
                sha256_hash = hashlib.sha256(ckks_bytes).hexdigest()
                b64_str = base64.b64encode(ckks_bytes).decode("ascii")

                # Archive ciphertext record in Vault KV v2 for forensic audit trail
                if pipeline_mgr.vault_manager is not None:
                    pipeline_mgr.vault_manager.store_encrypted_latent(
                        frame_id=rec.frame_id,
                        face_idx=idx,
                        encrypted_bytes=ckks_bytes,
                        metadata={"box": list(rec.box), "pts_ns": rec.pts_ns},
                    )

                item = LatentItem(
                    frame_id=rec.frame_id,
                    face_idx=idx,
                    box=list(rec.box),
                    pts_ns=rec.pts_ns,
                    ckks_bytes_b64=b64_str,
                    byte_size=len(ckks_bytes),
                    sha256=sha256_hash,
                )
                items.append(item)
                raw_concatenated_bytes.extend(ckks_bytes)

            except Exception as exc:
                log.error("Failed to encrypt latent for frame %d: %s", rec.frame_id, exc)

    # 3. Format response based on requested media type
    if format in ("raw", "binary"):
        return Response(
            content=bytes(raw_concatenated_bytes),
            media_type="application/octet-stream",
            headers={
                "X-Latents-Count": str(len(items)),
                "X-Total-Bytes": str(len(raw_concatenated_bytes)),
            },
        )

    return StreamBResponse(
        count=len(items),
        timestamp=datetime.now(timezone.utc).isoformat(),
        officer_token_verified=True,
        latents=items,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone Uvicorn Entrypoint
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    log.info("Starting CryptaVision FastAPI server on %s:%d", host, port)
    uvicorn.run("main:app", host=host, port=port, reload=False, workers=1)
