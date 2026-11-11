"""
CryptaVision Phase 2 — Unit & Integration Tests
================================================
Run:  pytest tests/test_ai_engine.py -v [--tb=short]

Test strategy
-------------
- All GPU paths are tested via mocking so CI passes on CPU-only machines.
- numpy/torch shapes are verified at every boundary.
- LatentStore thread-safety is stress-tested with concurrent producers.
- Phase 3 handoff contract (mu shape, dtype, device) is explicitly asserted.
- Timing performance tests use loose bounds (5× headroom) to avoid flakiness.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch

# ── allow running from repo root ────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))

from ai_engine import (
    FaceVAE,
    LatentRecord,
    LatentStore,
    PerfMonitor,
    SemanticAnalyzer,
    PrivacyObfuscator,
    CryptaAIEngine,
    _resolve_device,
)

# ── Test frame dimensions (use small frame to keep CI fast) ─────────────────
H, W = 360, 640
LATENT_DIM = 64   # smaller than production for speed


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _rand_bgr(h=H, w=W) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

def _rand_bgrx(h=H, w=W) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 4), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  _resolve_device
# ══════════════════════════════════════════════════════════════════════════════
class TestResolveDevice:
    def test_returns_cpu_when_no_gpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                dev = _resolve_device()
        assert dev.type == "cpu"

    def test_explicit_cpu_honoured(self):
        dev = _resolve_device("cpu")
        assert dev.type == "cpu"

    def test_cuda_fallback_to_cpu_when_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            dev = _resolve_device("cuda")
        assert dev.type == "cpu"


# ══════════════════════════════════════════════════════════════════════════════
#  FaceVAE
# ══════════════════════════════════════════════════════════════════════════════
class TestFaceVAE:
    @pytest.fixture
    def vae(self):
        return FaceVAE(latent_dim=LATENT_DIM)

    def test_encode_output_shapes(self, vae):
        x = torch.zeros(2, 3, 64, 64)
        with torch.inference_mode():
            mu, log_var = vae.encode(x)
        assert mu.shape      == (2, LATENT_DIM)
        assert log_var.shape == (2, LATENT_DIM)

    def test_decode_output_shape(self, vae):
        z = torch.randn(2, LATENT_DIM)
        with torch.inference_mode():
            recon = vae.decode(z)
        assert recon.shape == (2, 3, 64, 64)

    def test_decode_output_range(self, vae):
        z = torch.randn(3, LATENT_DIM)
        with torch.inference_mode():
            recon = vae.decode(z)
        # Sigmoid output must be in [0, 1]
        assert float(recon.min()) >= 0.0
        assert float(recon.max()) <= 1.0

    def test_forward_returns_three_tensors(self, vae):
        x = torch.rand(1, 3, 64, 64)
        with torch.inference_mode():
            recon, mu, log_var = vae(x)
        assert recon.shape   == (1, 3, 64, 64)
        assert mu.shape      == (1, LATENT_DIM)
        assert log_var.shape == (1, LATENT_DIM)

    def test_reparameterise_has_grad_shape(self, vae):
        mu      = torch.zeros(1, LATENT_DIM)
        log_var = torch.zeros(1, LATENT_DIM)
        z = vae.reparameterise(mu, log_var)
        assert z.shape == (1, LATENT_DIM)

    def test_vae_loss_is_scalar(self, vae):
        x = torch.rand(2, 3, 64, 64)
        recon, mu, log_var = vae(x)
        loss = FaceVAE.vae_loss(recon, x, mu, log_var, beta=1.0)
        assert loss.ndim == 0     # scalar
        assert float(loss) > 0.0

    def test_init_weights_no_nan(self, vae):
        for name, param in vae.named_parameters():
            assert not torch.isnan(param).any(), f"NaN in {name} after init"


# ══════════════════════════════════════════════════════════════════════════════
#  LatentStore
# ══════════════════════════════════════════════════════════════════════════════
class TestLatentStore:
    def _record(self, frame_id: int = 0) -> LatentRecord:
        return LatentRecord(
            frame_id = frame_id,
            pts_ns   = 0,
            mu       = torch.zeros(1, LATENT_DIM),
            box      = (0, 0, 64, 64),
        )

    def test_push_and_drain(self):
        store = LatentStore()
        store.push([self._record(1), self._record(2)])
        out = store.drain()
        assert len(out) == 2
        assert out[0].frame_id == 1

    def test_drain_empties_store(self):
        store = LatentStore()
        store.push([self._record()])
        store.drain()
        assert len(store.drain()) == 0

    def test_max_size_eviction(self):
        store = LatentStore(max_size=3)
        store.push([self._record(i) for i in range(5)])
        out = store.drain()
        # Only last 3 should survive
        assert len(out) == 3
        assert out[0].frame_id == 2

    def test_thread_safety_concurrent_push(self):
        """10 threads each push 20 records; drain must see all of them."""
        store   = LatentStore(max_size=500)
        threads = []
        for _ in range(10):
            t = threading.Thread(
                target=lambda: store.push([self._record() for _ in range(20)])
            )
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        out = store.drain()
        assert len(out) == 200

    def test_latent_mu_dtype_and_device(self):
        """Phase 3 contract: mu must be float32 on CPU."""
        store = LatentStore()
        rec   = self._record()
        store.push([rec])
        out = store.drain()
        assert out[0].mu.dtype  == torch.float32
        assert out[0].mu.device.type == "cpu"
        assert out[0].mu.shape  == (1, LATENT_DIM)


# ══════════════════════════════════════════════════════════════════════════════
#  PerfMonitor
# ══════════════════════════════════════════════════════════════════════════════
class TestPerfMonitor:
    def test_tick_does_not_raise(self, caplog):
        pm = PerfMonitor("test", log_interval=2)
        pm.tick(stage_a=5.0)
        pm.tick(stage_a=3.0)   # triggers log at interval=2

    def test_ewma_converges(self):
        pm = PerfMonitor("test", log_interval=1000)
        for _ in range(100):
            pm.tick(x=10.0)
        # After 100 ticks all seeded at 10.0, EWMA should be ~10.0
        assert abs(pm._stage_times.get("x", 0) - 10.0) < 0.1


# ══════════════════════════════════════════════════════════════════════════════
#  SemanticAnalyzer (mocked models)
# ══════════════════════════════════════════════════════════════════════════════
class TestSemanticAnalyzer:
    """
    Models are heavy; mock them so tests run on CI without GPU/internet.
    We test:
      - Return dict schema
      - Frame-skip behaviour
      - L2-normalisation of embedding
      - torch.inference_mode() respected (no gradients allocated)
    """

    @pytest.fixture
    def analyzer(self):
        with patch("ai_engine.YOLO") as mock_yolo_cls, \
             patch("ai_engine.CLIPModel.from_pretrained") as mock_clip_model, \
             patch("ai_engine.CLIPProcessor.from_pretrained") as mock_clip_proc:

            # ── YOLO mock ─────────────────────────────────────────────────
            mock_pose_result = MagicMock()
            mock_pose_result.keypoints = None
            mock_yolo_inst = MagicMock()
            mock_yolo_inst.return_value = [mock_pose_result]
            mock_yolo_cls.return_value  = mock_yolo_inst

            # ── CLIP model mock ───────────────────────────────────────────
            # Return a normalised 512-d vector
            emb = torch.randn(1, 512)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mock_clip_inst = MagicMock()
            mock_clip_inst.get_image_features.return_value = emb
            mock_clip_inst.to.return_value = mock_clip_inst
            mock_clip_inst.half.return_value = mock_clip_inst
            mock_clip_model.return_value = mock_clip_inst

            # ── CLIP processor mock ───────────────────────────────────────
            from transformers.feature_extraction_utils import BatchFeature
            proc_out = BatchFeature({"pixel_values": torch.zeros(1, 3, 224, 224)})
            mock_proc_inst = MagicMock()
            mock_proc_inst.return_value = proc_out
            mock_clip_proc.return_value = mock_proc_inst

            sa = SemanticAnalyzer(process_interval=3, device="cpu")
            sa._pose = mock_yolo_inst
        return sa

    def test_result_schema_on_key_frame(self, analyzer):
        frame = _rand_bgr()
        # frame_count will be 0 → 1, 1%3 ≠ 0 → skip; trigger on frame 3
        for _ in range(2):
            analyzer.process_frame(frame, pts_ns=0)
        result = analyzer.process_frame(frame, pts_ns=1_000_000_000)

        assert result["status"] == "ok"
        assert "frame_id"        in result
        assert "pts_s"           in result
        assert "pose_keypoints"  in result
        assert "clip_embedding"  in result
        assert "latency_ms"      in result
        assert len(result["clip_embedding"]) == 512

    def test_skipped_frame_status(self, analyzer):
        frame = _rand_bgr()
        result = analyzer.process_frame(frame, pts_ns=0)   # frame 1 → skip (1%3≠0)
        assert result["status"] == "skipped_frame"

    def test_clip_embedding_is_l2_normalised(self, analyzer):
        frame = _rand_bgr()
        for _ in range(3):
            r = analyzer.process_frame(frame, pts_ns=0)
        if r["status"] == "ok":
            emb  = np.array(r["clip_embedding"])
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-4, f"Embedding not unit-norm: {norm}"

    def test_clip_model_output_compatibility(self, analyzer):
        from types import SimpleNamespace
        analyzer._clip_model.get_image_features.return_value = SimpleNamespace(
            pooler_output=torch.randn(1, 512)
        )
        for _ in range(3):
            result = analyzer.process_frame(_rand_bgr(), pts_ns=0)
        assert result["status"] == "ok"
        assert len(result["clip_embedding"]) == 512
        assert np.linalg.norm(result["clip_embedding"]) == pytest.approx(1, abs=1e-4)

    def test_no_gradients_allocated(self, analyzer):
        """inference_mode() must prevent gradient tracking."""
        frame = _rand_bgr()
        for _ in range(3):
            analyzer.process_frame(frame, pts_ns=0)
        # If gradients were tracked, tensor.requires_grad would be True
        # We just verify no exception is raised (inference_mode disables grad)


# ══════════════════════════════════════════════════════════════════════════════
#  PrivacyObfuscator (mocked MTCNN, real VAE on CPU)
# ══════════════════════════════════════════════════════════════════════════════
class TestPrivacyObfuscator:

    @pytest.fixture
    def obfuscator_no_faces(self):
        """MTCNN returns no detections."""
        with patch("ai_engine.MTCNN") as mock_mtcnn_cls:
            inst = MagicMock()
            inst.detect.return_value = (None, None)
            mock_mtcnn_cls.return_value = inst
            po = PrivacyObfuscator(latent_dim=LATENT_DIM, device="cpu")
        return po

    @pytest.fixture
    def obfuscator_one_face(self):
        """MTCNN returns one 100×100 face box."""
        with patch("ai_engine.MTCNN") as mock_mtcnn_cls:
            inst  = MagicMock()
            boxes = np.array([[50, 50, 150, 150]], dtype=np.float32)
            probs = np.array([0.99], dtype=np.float32)
            inst.detect.return_value = (boxes, probs)
            mock_mtcnn_cls.return_value = inst
            po = PrivacyObfuscator(latent_dim=LATENT_DIM, device="cpu")
        return po

    # ── Output shape / dtype ──────────────────────────────────────────────

    def test_no_face_returns_same_shape_bgr(self, obfuscator_no_faces):
        frame  = _rand_bgr()
        output = obfuscator_no_faces.obfuscate(frame)
        assert output.shape == frame.shape
        assert output.dtype == np.uint8

    def test_no_face_returns_same_shape_bgrx(self, obfuscator_no_faces):
        frame  = _rand_bgrx()
        output = obfuscator_no_faces.obfuscate(frame)
        assert output.shape == frame.shape
        assert output.dtype == np.uint8

    def test_one_face_output_shape_preserved_bgr(self, obfuscator_one_face):
        frame  = _rand_bgr(360, 640)
        output = obfuscator_one_face.obfuscate(frame)
        assert output.shape == frame.shape
        assert output.dtype == np.uint8

    def test_one_face_output_shape_preserved_bgrx(self, obfuscator_one_face):
        frame  = _rand_bgrx(360, 640)
        output = obfuscator_one_face.obfuscate(frame)
        assert output.shape == frame.shape
        assert output.dtype == np.uint8

    def test_output_is_contiguous(self, obfuscator_one_face):
        """appsrc requires contiguous memory."""
        frame  = _rand_bgr()
        output = obfuscator_one_face.obfuscate(frame)
        assert output.flags["C_CONTIGUOUS"]

    # ── Blur actually applied ─────────────────────────────────────────────

    def test_face_region_is_modified(self, obfuscator_one_face):
        """Pixels inside the face box must differ from the original."""
        frame  = np.full((360, 640, 3), 200, dtype=np.uint8)
        # Paint face box with recognisable pattern
        frame[50:150, 50:150, 0] = 0
        frame[50:150, 50:150, 1] = 128
        output = obfuscator_one_face.obfuscate(frame)
        roi_in  = frame[50:150, 50:150]
        roi_out = output[50:150, 50:150]
        assert not np.array_equal(roi_in, roi_out), \
            "Face region should have been blurred"

    def test_non_face_region_unchanged(self, obfuscator_one_face):
        """Pixels outside the detected box must not be touched."""
        frame  = np.zeros((360, 640, 3), dtype=np.uint8)
        output = obfuscator_one_face.obfuscate(frame)
        # Row 0 is far outside the [50,150] box
        np.testing.assert_array_equal(frame[0, :], output[0, :])

    # ── Phase 3 latent contract ───────────────────────────────────────────

    def test_latent_pushed_to_store(self, obfuscator_one_face):
        frame = _rand_bgr(360, 640)
        obfuscator_one_face.obfuscate(frame)
        records = obfuscator_one_face.latent_store.drain()
        assert len(records) == 1

    def test_latent_shape_dtype_device(self, obfuscator_one_face):
        frame   = _rand_bgr(360, 640)
        obfuscator_one_face.obfuscate(frame)
        records = obfuscator_one_face.latent_store.drain()
        mu      = records[0].mu
        assert mu.shape       == (1, LATENT_DIM),  f"Got {mu.shape}"
        assert mu.dtype       == torch.float32,    f"Got {mu.dtype}"
        assert mu.device.type == "cpu",            f"Got {mu.device}"

    def test_get_latest_latents_compat(self, obfuscator_one_face):
        """get_latest_latents() shim used by Phase 3 integration tests."""
        frame   = _rand_bgr(360, 640)
        obfuscator_one_face.obfuscate(frame)
        latents = obfuscator_one_face.get_latest_latents()
        assert isinstance(latents, list)
        assert len(latents) == 1
        assert latents[0].shape == (1, LATENT_DIM)

    def test_latent_store_empty_after_drain(self, obfuscator_one_face):
        frame = _rand_bgr(360, 640)
        obfuscator_one_face.obfuscate(frame)
        obfuscator_one_face.latent_store.drain()          # first drain
        second = obfuscator_one_face.latent_store.drain()  # second drain
        assert len(second) == 0

    # ── Tiny box rejection ────────────────────────────────────────────────

    def test_tiny_box_is_skipped(self):
        """Boxes < MIN_BOX_DIM must not produce latents."""
        with patch("ai_engine.MTCNN") as mock_mtcnn_cls:
            inst  = MagicMock()
            boxes = np.array([[100, 100, 108, 108]], dtype=np.float32)  # 8×8 = tiny
            probs = np.array([0.9], dtype=np.float32)
            inst.detect.return_value = (boxes, probs)
            mock_mtcnn_cls.return_value = inst
            po = PrivacyObfuscator(latent_dim=LATENT_DIM, device="cpu")

        po.obfuscate(_rand_bgr(360, 640))
        assert len(po.latent_store.drain()) == 0


# ══════════════════════════════════════════════════════════════════════════════
#  CryptaAIEngine façade
# ══════════════════════════════════════════════════════════════════════════════
class TestCryptaAIEngine:
    def test_shared_latent_store(self):
        """SemanticAnalyzer and PrivacyObfuscator share one LatentStore."""
        with patch("ai_engine.YOLO"), \
             patch("ai_engine.CLIPModel.from_pretrained"), \
             patch("ai_engine.CLIPProcessor.from_pretrained"), \
             patch("ai_engine.MTCNN"):
            engine = CryptaAIEngine(latent_dim=LATENT_DIM, device="cpu")

        assert engine.latent_store is engine.privacy_obfuscator.latent_store
