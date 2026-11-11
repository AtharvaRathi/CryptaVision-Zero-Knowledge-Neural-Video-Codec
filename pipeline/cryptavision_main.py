#!/usr/bin/env python3
"""
CryptaVision — Phase 2 Integrated Main Entry Point
====================================================
Wires the Phase 2 CryptaAIEngine into the Phase 1 CryptaVisionPipeline
without modifying any Phase 1 source code.

Wiring strategy
---------------
Phase 1 exposes two internal processor slots:
  pipeline._semantic_processor  → obj with .process_frame(frame, pts_ns)
  pipeline._visual_obfuscator   → obj with .obfuscate(frame)

We replace both after construction, before calling pipeline.start().
This is the canonical extension point described in Phase 1's docstring.

Additionally, Phase 1's `on_semantic_frame` callback is connected to the
engine's analytics output sink so the result dict can be logged / forwarded
to any downstream consumer (e.g., a message queue for Phase 3).
"""

import json
import logging
import signal
import sys
from pathlib import Path
from typing import Optional, Any

import numpy as np

# Add project root to sys.path so crypto_engine and vault_manager are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryptavision_pipeline import CryptaVisionPipeline
from ai_engine import CryptaAIEngine, LatentRecord

try:
    from crypto_engine import CryptoEngine, TENSEAL_AVAILABLE
    from vault_manager import VaultManager, VaultConfig
    CRYPTO_AVAILABLE = TENSEAL_AVAILABLE
except ImportError:
    CryptoEngine = None  # type: ignore
    VaultManager = None  # type: ignore
    CRYPTO_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("CryptaVision.Main")


# ══════════════════════════════════════════════════════════════════════════════
#  Semantic result sink — replace with Kafka/Redis producer in production
# ══════════════════════════════════════════════════════════════════════════════
def semantic_result_sink(result: dict) -> None:
    """
    Receives the output dict from SemanticAnalyzer.process_frame().
    Stub: logs a summary line.  Replace with:
        - Kafka producer (confluent-kafka)
        - Redis XADD to a stream
        - gRPC call to Feature Store
    """
    if result.get("status") == "skipped_frame":
        return
    log.info(
        "SEMANTIC | frame=%d  pts=%.3fs  humans=%d  "
        "pose_kps=%d  emb_dim=%d  latency=%s",
        result["frame_id"],
        result["pts_s"],
        result["num_humans"],
        sum(len(p) for p in result["pose_keypoints"]),
        len(result["clip_embedding"]),
        result["latency_ms"],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3 handoff adapter — stub for crypto consumer
# ══════════════════════════════════════════════════════════════════════════════
def phase3_latent_consumer(
    engine: CryptaAIEngine,
    crypto_engine: Optional[Any] = None,
    vault_manager: Optional[Any] = None,
) -> None:
    """
    Drains the LatentStore and executes Phase 3:
      1. Encrypts [1, 128] FP32 face embeddings using TenSEAL CKKS (scale=2^40).
      2. Archives serialized ciphertexts to HashiCorp Vault KV v2.
    """
    records = engine.latent_store.drain()
    if not records:
        return

    log.info(
        "LATENT STORE | Processing %d face latent(s) for Phase 3 crypto layer.",
        len(records),
    )

    for idx, rec in enumerate(records):
        if crypto_engine is not None and vault_manager is not None:
            try:
                # 1. Homomorphic Encryption via TenSEAL CKKS
                encrypted_bytes = crypto_engine.encrypt_latent(rec.mu)

                # 2. Archive to HashiCorp Vault KV v2
                vault_path = vault_manager.store_encrypted_latent(
                    frame_id=rec.frame_id,
                    face_idx=idx,
                    encrypted_bytes=encrypted_bytes,
                    metadata={"box": rec.box, "pts_ns": rec.pts_ns},
                )
                log.info(
                    "HE-ENCRYPTED & ARCHIVED | frame=%d face=%d ciphertext_size=%d bytes path=%s",
                    rec.frame_id,
                    idx,
                    len(encrypted_bytes),
                    vault_path,
                )
            except Exception as exc:
                log.error("Failed to encrypt/archive latent for frame %d: %s", rec.frame_id, exc)
        else:
            log.debug(
                "  frame=%d  box=%s  mu_norm=%.4f (crypto engine inactive)",
                rec.frame_id,
                rec.box,
                float(rec.mu.norm()),
            )


# ══════════════════════════════════════════════════════════════════════════════
#  Adapter shim: SemanticAnalyzer → Phase 1 SemanticProcessor interface
# ══════════════════════════════════════════════════════════════════════════════
class _SemanticProcessorShim:
    """
    Wraps SemanticAnalyzer so it satisfies the Phase 1 interface:
        .process_frame(frame_np: np.ndarray, pts_ns: int) -> np.ndarray

    Phase 1 calls this and discards the return value (it uses the
    on_semantic_frame callback for downstream delivery).
    The shim forwards results to the result_sink callable.
    """

    def __init__(self, analyzer, result_sink=None):
        self._analyzer    = analyzer
        self._result_sink = result_sink or (lambda r: None)

    def process_frame(self, frame_np: np.ndarray, pts_ns: int):
        result = self._analyzer.process_frame(frame_np, pts_ns)
        self._result_sink(result)
        # Phase 1 ignores the return but expects ndarray; return empty vector
        return np.array([], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="CryptaVision Phase 2 — Production AI Pipeline"
    )
    parser.add_argument(
        "--source",
        default="rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4",
        help="RTSP URL or local .mp4/.mkv path",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Seconds to run (0 = until EOS)",
    )
    parser.add_argument(
        "--interval", type=int, default=5,
        help="Semantic extraction frame-skip interval (default: 5)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device override: 'cuda', 'cpu', 'mps'",
    )
    parser.add_argument(
        "--yolo-weights", default="yolov8n-pose.pt",
        help="Path to YOLO pose weights file",
    )
    parser.add_argument(
        "--vae-weights", default=None,
        help="Path to FaceVAE .pt checkpoint (omit for random init)",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=128,
        help="VAE latent dimension (must match trained checkpoint)",
    )
    parser.add_argument(
        "--enable-crypto", action="store_true", default=True,
        help="Enable Phase 3 TenSEAL CKKS Homomorphic Encryption and Vault storage (default: True)",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── 1. Construct Phase 2 AI engine ───────────────────────────────────
    log.info("Initialising Phase 2 AI Engine…")
    engine = CryptaAIEngine(
        process_interval = args.interval,
        yolo_weights     = args.yolo_weights,
        vae_weights      = args.vae_weights,
        latent_dim       = args.latent_dim,
        device           = args.device,
    )

    # ── 2. Construct Phase 3 Crypto & Vault (if enabled) ─────────────────
    crypto_engine = None
    vault_manager = None
    if args.enable_crypto and CRYPTO_AVAILABLE and CryptoEngine is not None and VaultManager is not None:
        try:
            log.info("Initializing Phase 3 TenSEAL CKKS Homomorphic Encryption Engine...")
            crypto_engine = CryptoEngine(latent_dim=args.latent_dim)
            vault_manager = VaultManager()
            # Secure the secret context into Vault KV v2
            vault_manager.store_secret_context(crypto_engine.serialize_secret_context())
            log.info("Phase 3 Cryptographic Layer & Vault Key Management ACTIVE.")
        except Exception as exc:
            log.warning("Phase 3 Crypto initialization deferred or failed: %s", exc)

    # ── 3. Construct Phase 1 pipeline ────────────────────────────────────
    pipeline = CryptaVisionPipeline(source_uri=args.source)

    # ── 4. Inject Phase 2 processors into Phase 1 slots ──────────────────
    pipeline._semantic_processor = _SemanticProcessorShim(
        analyzer     = engine.semantic_analyzer,
        result_sink  = semantic_result_sink,
    )
    pipeline._visual_obfuscator = engine.privacy_obfuscator

    log.info("Phase 2 processors injected into Phase 1 pipeline slots.")

    # ── 5. SIGINT / SIGTERM handler ───────────────────────────────────────
    def _stop(signum, frame):
        log.info("Signal %d received — stopping pipeline…", signum)
        pipeline.stop()

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    # ── 6. Start pipeline ─────────────────────────────────────────────────
    pipeline.start()

    # ── 7. Main loop: drain latent store periodically ────────────────────
    try:
        import time
        end_time = (time.time() + args.duration) if args.duration > 0 else None

        while True:
            # Forward latents to Phase 3 consumer every 0.5 s
            phase3_latent_consumer(engine, crypto_engine=crypto_engine, vault_manager=vault_manager)

            if pipeline._eos_event.is_set():
                break
            if end_time and time.time() >= end_time:
                log.info("Duration elapsed — stopping.")
                break

            time.sleep(0.5)
    finally:
        pipeline.stop()
        # Final drain so no latents are lost on shutdown
        phase3_latent_consumer(engine, crypto_engine=crypto_engine, vault_manager=vault_manager)

    return 0


if __name__ == "__main__":
    sys.exit(main())
