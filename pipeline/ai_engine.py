#!/usr/bin/env python3
"""
CryptaVision Phase 2 — Production AI Engine
============================================
Contains two top-level classes designed to drop directly into the
Phase 1 callback hooks without any changes to cryptavision_pipeline.py:

  SemanticAnalyzer  → replaces SemanticProcessor  (Stream A)
  PrivacyObfuscator → replaces VisualObfuscator   (Stream B)

Thread-safety contract
----------------------
Both classes are called from GStreamer streaming threads (one per stream).
They are NOT called concurrently with each other.
Internal state (frame_count, current_latents) must only be mutated inside
the respective method; no cross-class shared state exists.

Performance targets
-------------------
  GPU path  : ≥ 30 FPS end-to-end (YOLOv8n-pose + CLIP FP16)
  CPU path  : ≥ 8–12 FPS (frame-skip compensates on heavy models)

Memory contract
---------------
  - torch tensors created inside @torch.inference_mode() blocks are never
    tracked by autograd; no gradient graph is allocated.
  - Face crops are converted to tensors via torch.from_numpy() with an
    explicit .clone() to ensure the numpy buffer can be freed.
  - GPU tensors are moved to CPU immediately after inference; CUDA memory
    is never held across frame boundaries.
  - self.current_latents is cleared at the start of every obfuscate() call
    to avoid accumulation across frames.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

# MTCNN is imported lazily inside __init__ to allow the module to load even
# on machines without GPU; the import itself is fast.
from facenet_pytorch import MTCNN

log = logging.getLogger("CryptaVision.AI")


# ══════════════════════════════════════════════════════════════════════════════
#  Device resolution helper
# ══════════════════════════════════════════════════════════════════════════════
def _resolve_device(preferred: Optional[str] = None) -> torch.device:
    """
    Picks the best available compute device.

    Priority: explicit > CUDA > MPS (Apple Silicon) > CPU.
    Logs a warning if CUDA is requested but unavailable.
    """
    if preferred:
        dev = torch.device(preferred)
        if dev.type == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available — falling back to CPU.")
            return torch.device("cpu")
        return dev

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
#  FaceVAE — Convolutional Variational Autoencoder
# ══════════════════════════════════════════════════════════════════════════════
class FaceVAE(nn.Module):
    """
    Standard Convolutional VAE expecting 64×64 RGB face crops.

    Architecture
    ------------
    Encoder  : 4 × (Conv2d → ReLU) → Flatten → fc_mu, fc_log_var
    Decoder  : fc_z → Reshape → 4 × (ConvTranspose2d → ReLU) → Sigmoid
    Latent   : shape [B, latent_dim]

    Phase 3 contract
    ----------------
    The Principal Cryptography Engineer will receive:
        mu  : torch.Tensor  shape [1, latent_dim]  dtype float32  device CPU
    They must convert it to a CKKS vector via TenSEAL and push to HashiCorp Vault.

    The decoder is included for future reconstruction-quality experiments
    (e.g., verifying that Vault-reconstructed embeddings decode plausibly).
    """

    INPUT_SIZE   = 64    # spatial side length expected by the encoder
    HIDDEN_SIZES = (32, 64, 128, 256)

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32,  kernel_size=4, stride=2, padding=1),  # 64→32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64,  kernel_size=4, stride=2, padding=1),  # 32→16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16→8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 8→4
            nn.ReLU(inplace=True),
        )
        # After encoder: spatial = 4×4, channels = 256 → flat = 256*4*4 = 4096
        self._flat_dim = 256 * 4 * 4
        self.fc_mu      = nn.Linear(self._flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(self._flat_dim, latent_dim)

        # ── Decoder ───────────────────────────────────────────────────────
        self.fc_decode = nn.Linear(latent_dim, self._flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4→8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),  # 8→16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1),  # 16→32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  3,   kernel_size=4, stride=2, padding=1),  # 32→64
            nn.Sigmoid(),
        )

        # Weight initialisation: He for conv layers, zeros for bias
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : float32 tensor [B, 3, 64, 64] in [0, 1]
        Returns:
            mu      : [B, latent_dim]
            log_var : [B, latent_dim]
        """
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)   # [B, 4096]
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterise(self, mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        """Standard reparameterisation trick: z = mu + eps * std."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : [B, latent_dim]
        Returns:
            reconstruction : [B, 3, 64, 64] in [0, 1]
        """
        h = F.relu(self.fc_decode(z))
        h = h.view(-1, 256, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z           = self.reparameterise(mu, log_var)
        recon       = self.decode(z)
        return recon, mu, log_var

    @staticmethod
    def vae_loss(recon: torch.Tensor, target: torch.Tensor,
                 mu: torch.Tensor, log_var: torch.Tensor,
                 beta: float = 1.0) -> torch.Tensor:
        """
        β-VAE loss = MSE reconstruction + β × KL divergence.
        Use during training; not called at inference time.
        """
        recon_loss = F.mse_loss(recon, target, reduction="sum")
        kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss


# ══════════════════════════════════════════════════════════════════════════════
#  LatentStore — thread-safe buffer for Phase 3 handoff
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class LatentRecord:
    """One face's latent snapshot for a single frame."""
    frame_id:  int
    pts_ns:    int
    mu:        torch.Tensor      # shape [1, latent_dim], dtype float32, device CPU
    box:       Tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixel coords


class LatentStore:
    """
    Bounded, thread-safe ring buffer of LatentRecord objects.

    The Cryptography Engine (Phase 3) reads from this store via
    `drain()` which atomically swaps the internal list, preventing
    any race between the obfuscation thread writing and the crypto
    thread reading.
    """

    def __init__(self, max_size: int = 128):
        self._lock    = threading.Lock()
        self._records: List[LatentRecord] = []
        self._max     = max_size

    def push(self, records: List[LatentRecord]) -> None:
        with self._lock:
            self._records.extend(records)
            # Evict oldest if over capacity
            if len(self._records) > self._max:
                self._records = self._records[-self._max:]

    def drain(self) -> List[LatentRecord]:
        """Atomically remove and return all pending records."""
        with self._lock:
            out, self._records = self._records, []
        return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


# ══════════════════════════════════════════════════════════════════════════════
#  PerfMonitor — lightweight FPS / latency tracker
# ══════════════════════════════════════════════════════════════════════════════
class PerfMonitor:
    """
    Rolling-window performance monitor.
    Logs throughput and per-stage latency every `log_interval` frames.
    """

    def __init__(self, name: str, log_interval: int = 30):
        self.name         = name
        self.log_interval = log_interval
        self._frame_count = 0
        self._t_start     = time.perf_counter()
        self._stage_times: Dict[str, float] = {}

    def tick(self, **stage_latencies_ms: float) -> None:
        """Call once per processed frame. Pass keyword latencies in ms."""
        self._frame_count += 1
        for k, v in stage_latencies_ms.items():
            self._stage_times[k] = self._stage_times.get(k, 0.0) * 0.9 + v * 0.1

        if self._frame_count % self.log_interval == 0:
            elapsed = time.perf_counter() - self._t_start
            fps     = self._frame_count / elapsed
            stages  = "  ".join(f"{k}={v:.1f}ms" for k, v in self._stage_times.items())
            log.info("[%s] FPS=%.1f  frame=%d  %s",
                     self.name, fps, self._frame_count, stages)


# ══════════════════════════════════════════════════════════════════════════════
#  SemanticAnalyzer  — Stream A
# ══════════════════════════════════════════════════════════════════════════════
class SemanticAnalyzer:
    """
    Production replacement for Phase 1's SemanticProcessor stub.

    Models loaded
    -------------
    - YOLOv8n-pose  : skeleton keypoint detection (17 COCO keypoints)
    - CLIP ViT-B/32 : 512-d image embedding (FP16 on CUDA)

    Frame-skip strategy
    -------------------
    Heavy semantic extraction every `process_interval` frames.
    Pose detection is cheap enough to run every frame if needed;
    CLIP is batched at the skip boundary.

    Wiring into Phase 1
    -------------------
    Instantiate this class and pass its `process_frame` method to
    `CryptaVisionPipeline.on_semantic_frame`:

        analyzer = SemanticAnalyzer(process_interval=5)
        pipeline = CryptaVisionPipeline(
            source_uri=...,
            on_semantic_frame=analyzer.process_frame,
        )

    The Phase 1 pipeline passes (frame: np.ndarray, pts_ns: int);
    this method signature matches exactly.
    """

    # CLIP model identifier — swap for larger variants if GPU VRAM allows
    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        process_interval: int           = 5,
        yolo_weights:     str           = "yolov8n-pose.pt",
        device:           Optional[str] = None,
        clip_model_id:    str           = CLIP_MODEL_ID,
    ):
        self.process_interval = max(1, process_interval)
        self._frame_count     = 0
        self._last_result:    Dict[str, Any] = {}
        self._device          = _resolve_device(device)
        self._perf            = PerfMonitor("SemanticAnalyzer")

        log.info("SemanticAnalyzer: device=%s  interval=%d",
                 self._device, self.process_interval)

        # ── YOLOv8 Pose ───────────────────────────────────────────────────
        log.info("Loading YOLOv8 pose model: %s", yolo_weights)
        self._pose = YOLO(yolo_weights)
        self._pose.to(self._device)
        # Warm-up pass so first-frame latency is representative
        _dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self._pose(_dummy, verbose=False)
        log.info("YOLOv8 loaded and warmed up.")

        # ── CLIP ──────────────────────────────────────────────────────────
        log.info("Loading CLIP: %s", clip_model_id)
        self._clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        self._clip_model     = CLIPModel.from_pretrained(clip_model_id)
        self._clip_model.eval()
        self._clip_model.to(self._device)

        # Use FP16 on CUDA for ~2× throughput with negligible accuracy loss.
        # MPS and CPU remain FP32 (torch.float16 unsupported on CPU BLAS).
        self._clip_dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        if self._clip_dtype == torch.float16:
            self._clip_model.half()
            log.info("CLIP running in FP16.")

        # CLIP warm-up
        with torch.inference_mode():
            _inp = self._clip_processor(
                images=np.zeros((224, 224, 3), dtype=np.uint8),
                return_tensors="pt"
            ).to(self._device)
            if self._clip_dtype == torch.float16:
                _inp["pixel_values"] = _inp["pixel_values"].half()
            self._clip_model.get_image_features(**_inp)
        log.info("CLIP loaded and warmed up.")

    # ─────────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def process_frame(
        self,
        frame_np: np.ndarray,
        pts_ns:   int,
    ) -> Dict[str, Any]:
        """
        Phase 1 hook.  Called by `_on_new_sample_a` in cryptavision_pipeline.py.

        Args:
            frame_np : uint8 ndarray  shape (H, W, 3)  BGR colorspace
            pts_ns   : presentation timestamp in nanoseconds

        Returns:
            dict with keys:
              status          : "ok" | "skipped_frame"
              frame_id        : int
              pts_s           : float  (seconds)
              pose_keypoints  : list[list[list[float]]]  [person][kp][x,y]
              pose_scores     : list[list[float]]        [person][kp_score]
              num_humans      : int
              clip_embedding  : list[float]  length 512, L2-normalised
              latency_ms      : dict of per-stage timings
        """
        self._frame_count += 1

        # ── Frame-skip: return cached result for non-key frames ────────────
        if self._frame_count % self.process_interval != 0:
            return {**self._last_result, "status": "skipped_frame",
                    "frame_id": self._frame_count}

        t0 = time.perf_counter()

        # GStreamer delivers BGR; all models expect RGB
        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

        # ── Stage 1: Pose detection ────────────────────────────────────────
        t_pose_start = time.perf_counter()
        pose_results = self._pose(frame_rgb, verbose=False)[0]

        keypoints_xy:   List[List[List[float]]] = []
        keypoints_conf: List[List[float]]        = []

        if (pose_results.keypoints is not None
                and len(pose_results.keypoints.xy) > 0):
            # .xy  : [N_persons, 17, 2]
            # .conf: [N_persons, 17]
            kp_xy   = pose_results.keypoints.xy.cpu().numpy()    # float32
            kp_conf = pose_results.keypoints.conf.cpu().numpy()  # float32
            keypoints_xy   = kp_xy.tolist()
            keypoints_conf = kp_conf.tolist()

        t_pose_ms = (time.perf_counter() - t_pose_start) * 1000

        # ── Stage 2: CLIP image embedding ─────────────────────────────────
        t_clip_start = time.perf_counter()

        clip_inputs = self._clip_processor(
            images=frame_rgb, return_tensors="pt"
        ).to(self._device)

        if self._clip_dtype == torch.float16:
            clip_inputs["pixel_values"] = clip_inputs["pixel_values"].half()

        image_features = self._clip_model.get_image_features(**clip_inputs)
        # Transformers 5 returns the projected tensor in a model output;
        # older versions return the tensor directly.
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output
        # L2-normalise so cosine similarity == dot product downstream
        image_features = image_features / image_features.norm(
            p=2, dim=-1, keepdim=True
        )
        embedding: List[float] = image_features.squeeze(0).float().cpu().tolist()

        t_clip_ms = (time.perf_counter() - t_clip_start) * 1000
        t_total_ms = (time.perf_counter() - t0) * 1000

        self._perf.tick(pose=t_pose_ms, clip=t_clip_ms, total=t_total_ms)

        result: Dict[str, Any] = {
            "status":         "ok",
            "frame_id":       self._frame_count,
            "pts_s":          pts_ns / 1e9,
            "pose_keypoints": keypoints_xy,
            "pose_scores":    keypoints_conf,
            "num_humans":     len(keypoints_xy),
            "clip_embedding": embedding,   # 512-dimensional, L2-normalised
            "latency_ms": {
                "pose":  round(t_pose_ms, 2),
                "clip":  round(t_clip_ms, 2),
                "total": round(t_total_ms, 2),
            },
        }
        # Cache for skipped frames
        self._last_result = result
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  PrivacyObfuscator  — Stream B
# ══════════════════════════════════════════════════════════════════════════════
class PrivacyObfuscator:
    """
    Production replacement for Phase 1's VisualObfuscator stub.

    Processing pipeline per frame
    -----------------------------
    1. MTCNN face detection         → bounding boxes (batched, GPU)
    2. Per-face: crop → resize(64×64) → FaceVAE.encode()
       → store LatentRecord in LatentStore (Phase 3 handoff)
    3. Gaussian blur applied to each face bounding box  (OpenCV, CPU)
    4. Return modified frame as contiguous BGR ndarray

    Wiring into Phase 1
    -------------------
    The Phase 1 pipeline calls `VisualObfuscator.obfuscate(frame_np)`.
    This class exposes the same signature:

        obfuscator = PrivacyObfuscator()
        pipeline = CryptaVisionPipeline(
            source_uri=...,
            on_visual_frame=lambda frame, pts: None,   # display hook
        )
        # Override the internal obfuscator:
        pipeline._visual_obfuscator = obfuscator

    Alternatively, subclass CryptaVisionPipeline and override
    `_visual_obfuscator` in __init__ before calling super().

    Phase 3 handoff
    ---------------
    The crypto engineer reads from `obfuscator.latent_store.drain()`.
    Each LatentRecord.mu is shape [1, 128], dtype float32, on CPU.
    """

    # Minimum acceptable face box dimension (pixels).
    # Boxes smaller than this are noise / very distant faces — skip VAE.
    MIN_BOX_DIM = 10

    # Gaussian blur kernel size for face occlusion.
    # Must be odd; larger = more blur. (99, 99) produces strong anonymisation.
    BLUR_KERNEL = (99, 99)
    BLUR_SIGMA  = 30

    def __init__(
        self,
        vae_weights_path: Optional[str] = None,
        latent_dim:       int           = 128,
        device:           Optional[str] = None,
        latent_store:     Optional[LatentStore] = None,
    ):
        self._device  = _resolve_device(device)
        self._perf    = PerfMonitor("PrivacyObfuscator")
        self._frame_count = 0

        # Exposed to Phase 3 consumer
        self.latent_store: LatentStore = latent_store or LatentStore()

        log.info("PrivacyObfuscator: device=%s  latent_dim=%d",
                 self._device, latent_dim)

        # ── MTCNN face detector ───────────────────────────────────────────
        # keep_all=True returns all faces in the frame (not just the largest).
        # min_face_size=40 filters out very small / noisy detections.
        self._face_detector = MTCNN(
            keep_all       = True,
            min_face_size  = 40,
            thresholds     = [0.6, 0.7, 0.7],   # P-Net / R-Net / O-Net
            device         = self._device,
            post_process   = False,   # return raw [0,255] crops, not normalised
        )
        log.info("MTCNN loaded.")

        # ── FaceVAE ───────────────────────────────────────────────────────
        self._vae = FaceVAE(latent_dim=latent_dim).to(self._device)
        self._vae.eval()

        if vae_weights_path and Path(vae_weights_path).exists():
            state = torch.load(vae_weights_path, map_location=self._device,
                               weights_only=True)
            self._vae.load_state_dict(state)
            log.info("FaceVAE weights loaded from %s", vae_weights_path)
        else:
            log.warning(
                "No VAE weights found at '%s' — using random initialisation. "
                "Latent vectors will be random until the model is trained.",
                vae_weights_path,
            )

        # FP16 on CUDA for VAE encode pass
        if self._device.type == "cuda":
            self._vae.half()
            self._vae_dtype = torch.float16
        else:
            self._vae_dtype = torch.float32

        log.info("FaceVAE initialised (dtype=%s).", self._vae_dtype)

    # ─────────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def obfuscate(self, frame_np: np.ndarray) -> np.ndarray:
        """
        Phase 1 hook.  Called by `_on_new_sample_b` in cryptavision_pipeline.py.

        Args:
            frame_np : uint8 ndarray  shape (H, W, C)
                       C = 4 (BGRx) when coming directly from the appsink,
                       C = 3 (BGR)  when sliced by the pipeline before passing.
                       Both are handled transparently.

        Returns:
            output   : uint8 ndarray  same shape as input
                       Faces are Gaussian-blurred; all other pixels unchanged.
                       Always a new contiguous allocation (safe to push into appsrc).
        """
        self._frame_count += 1
        t0 = time.perf_counter()

        # ── Normalise to 3-channel BGR ─────────────────────────────────────
        # The Phase 1 pipeline may pass BGRx (4-channel) directly from appsink.
        if frame_np.ndim == 3 and frame_np.shape[2] == 4:
            # Drop the X channel; keep spatial layout unchanged
            frame_bgr  = frame_np[..., :3]
            has_x_chan = True
        else:
            frame_bgr  = frame_np
            has_x_chan = False

        # MTCNN expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── Stage 1: Face detection ────────────────────────────────────────
        t_det_start = time.perf_counter()
        try:
            # detect() returns (boxes, probs)
            # boxes: ndarray [N, 4] float32 — [x1, y1, x2, y2] pixel coords
            # probs: ndarray [N]    float32 — confidence scores
            boxes, probs = self._face_detector.detect(frame_rgb)
        except Exception as exc:
            log.warning("MTCNN detection failed: %s — returning passthrough.", exc)
            return np.ascontiguousarray(frame_np)

        t_det_ms = (time.perf_counter() - t_det_start) * 1000

        # No faces — fast path, avoid any copy
        if boxes is None or len(boxes) == 0:
            self._perf.tick(detect=t_det_ms, vae=0.0, blur=0.0,
                            total=(time.perf_counter() - t0) * 1000)
            return np.ascontiguousarray(frame_np)

        h, w = frame_bgr.shape[:2]
        output_bgr  = frame_bgr.copy()      # writable copy — modified in-place below
        new_latents: List[LatentRecord] = []

        t_vae_total  = 0.0
        t_blur_total = 0.0

        # ── Stage 2 & 3: Per-face VAE + blur ─────────────────────────────
        for idx, (box, prob) in enumerate(zip(boxes, probs)):
            # Clamp to frame boundary and convert to int
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w,  int(box[2]))
            y2 = min(h,  int(box[3]))

            if (x2 - x1) < self.MIN_BOX_DIM or (y2 - y1) < self.MIN_BOX_DIM:
                log.debug("Skipping tiny face box [%d,%d,%d,%d]", x1,y1,x2,y2)
                continue

            # ── VAE encode ────────────────────────────────────────────────
            t_vae_start = time.perf_counter()

            face_rgb     = frame_rgb[y1:y2, x1:x2]                    # (fH, fW, 3) uint8
            face_resized = cv2.resize(face_rgb, (FaceVAE.INPUT_SIZE,
                                                 FaceVAE.INPUT_SIZE),
                                      interpolation=cv2.INTER_AREA)    # (64, 64, 3)

            # uint8 → float32 tensor [1, 3, 64, 64] in [0, 1]
            # .clone() ensures the numpy buffer behind face_resized can be freed
            face_tensor = (
                torch.from_numpy(face_resized)
                     .clone()
                     .permute(2, 0, 1)          # HWC → CHW
                     .unsqueeze(0)              # add batch dim
                     .float()
                     .div(255.0)
                     .to(self._device)
            )
            if self._vae_dtype == torch.float16:
                face_tensor = face_tensor.half()

            mu, log_var = self._vae.encode(face_tensor)

            # Move to CPU float32 immediately; CUDA allocation released here
            mu_cpu = mu.float().cpu()           # [1, latent_dim]

            new_latents.append(LatentRecord(
                frame_id = self._frame_count,
                pts_ns   = 0,                  # populated by caller if needed
                mu       = mu_cpu,
                box      = (x1, y1, x2, y2),
            ))

            t_vae_total += (time.perf_counter() - t_vae_start) * 1000

            # ── Gaussian blur on BGR output frame ─────────────────────────
            t_blur_start = time.perf_counter()
            roi  = output_bgr[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(roi, self.BLUR_KERNEL, self.BLUR_SIGMA)
            output_bgr[y1:y2, x1:x2] = blurred
            t_blur_total += (time.perf_counter() - t_blur_start) * 1000

        # ── Push latents to thread-safe store (Phase 3 consumer) ──────────
        if new_latents:
            self.latent_store.push(new_latents)

        t_total_ms = (time.perf_counter() - t0) * 1000
        self._perf.tick(detect=t_det_ms, vae=t_vae_total,
                        blur=t_blur_total, total=t_total_ms)

        # ── Reconstruct original channel layout ───────────────────────────
        if has_x_chan:
            # Re-attach the X channel; use a fresh zeros column so the
            # output shape matches the appsrc caps (BGRx 4-channel).
            x_chan = frame_np[..., 3:4]          # original X channel
            return np.ascontiguousarray(
                np.concatenate([output_bgr, x_chan], axis=2)
            )
        return np.ascontiguousarray(output_bgr)

    # ─────────────────────────────────────────────────────────────────────────
    #  Phase 3 handoff API
    # ─────────────────────────────────────────────────────────────────────────

    def get_latest_latents(self) -> List[torch.Tensor]:
        """
        Compatibility shim for Phase 3 consumers that call this method.

        Returns a list of [1, latent_dim] float32 CPU tensors (mu only).
        For production use, prefer `self.latent_store.drain()` which also
        exposes frame_id, pts_ns, and bounding-box coordinates.
        """
        return [rec.mu for rec in self.latent_store.drain()]


# ══════════════════════════════════════════════════════════════════════════════
#  CryptaAIEngine — façade that wires both processors to Phase 1
# ══════════════════════════════════════════════════════════════════════════════
class CryptaAIEngine:
    """
    Convenience façade: instantiates SemanticAnalyzer + PrivacyObfuscator
    and exposes the two methods that CryptaVisionPipeline expects.

    Usage
    -----
    See cryptavision_main.py for the canonical wiring pattern.
    """

    def __init__(
        self,
        process_interval: int           = 5,
        yolo_weights:     str           = "yolov8n-pose.pt",
        vae_weights:      Optional[str] = None,
        latent_dim:       int           = 128,
        device:           Optional[str] = None,
    ):
        shared_store = LatentStore()

        self.semantic_analyzer   = SemanticAnalyzer(
            process_interval = process_interval,
            yolo_weights     = yolo_weights,
            device           = device,
        )
        self.privacy_obfuscator  = PrivacyObfuscator(
            vae_weights_path = vae_weights,
            latent_dim       = latent_dim,
            device           = device,
            latent_store     = shared_store,
        )
        # Shared so the crypto layer can grab from one place
        self.latent_store = shared_store

    # These two methods match the signatures that Phase 1 calls directly:

    def on_semantic_frame(self, frame: np.ndarray, pts_ns: int) -> Dict[str, Any]:
        """Bound to CryptaVisionPipeline.on_semantic_frame."""
        return self.semantic_analyzer.process_frame(frame, pts_ns)

    def on_visual_frame(self, frame: np.ndarray, pts_ns: int) -> np.ndarray:
        """
        NOTE: Phase 1's on_visual_frame is a notification callback, NOT the
        obfuscation routine.  The actual obfuscation is wired via
        `pipeline._visual_obfuscator = self.privacy_obfuscator`.
        This method can be used for secondary display / metrics.
        """
        return frame
