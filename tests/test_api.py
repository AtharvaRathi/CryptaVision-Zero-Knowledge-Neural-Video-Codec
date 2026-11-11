"""
CryptaVision Phase 4 — FastAPI Orchestration Layer Tests
========================================================
Validates:
1. Health checks & telemetry (/health, /stream/status).
2. Asynchronous background pipeline startup (/stream/start, /stream/stop).
3. Server-Sent Events (SSE) streaming for Stream A (/analytics/stream-a).
4. Protected forensic decryption endpoint for Stream B (/analytics/stream-b)
   with Vault compliance token verification and serialized CKKS ciphertexts.

Run with:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from main import app, broadcaster, pipeline_mgr
from ai_engine import LatentRecord, LatentStore


@pytest.fixture
def client():
    """Provides a synchronous FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client


# ══════════════════════════════════════════════════════════════════════════════
#  Health & Telemetry Tests
# ══════════════════════════════════════════════════════════════════════════════
def test_health_check(client: TestClient):
    """Verifies that the /health endpoint responds with operational diagnostics."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "gpu_available" in data
    assert "pipeline_state" in data


def test_stream_status_idle(client: TestClient):
    """Verifies stream telemetry when no pipeline is running."""
    response = client.get("/stream/status")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] in ("IDLE", "STOPPED")
    assert "uptime_seconds" in data
    assert "active_stream_a_subscribers" in data


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline Control Tests (/stream/start, /stream/stop)
# ══════════════════════════════════════════════════════════════════════════════
def test_stream_start_validation(client: TestClient):
    """Validates request schema rejects malformed requests."""
    # Empty payload
    resp = client.post("/stream/start", json={})
    # Should use defaults or succeed
    assert resp.status_code == 200

    # Invalid process_interval (< 1)
    bad_resp = client.post("/stream/start", json={"process_interval": -5})
    assert bad_resp.status_code == 422


@patch.object(pipeline_mgr, "start_pipeline")
def test_stream_start_dispatched(mock_start, client: TestClient):
    """Verifies pipeline start is dispatched asynchronously."""
    mock_start.return_value = None

    payload = {
        "source_uri": "rtsp://10.0.0.1/live",
        "process_interval": 3,
        "latent_dim": 128,
        "enable_crypto": True,
    }
    response = client.post("/stream/start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "initiating"
    assert data["source_uri"] == "rtsp://10.0.0.1/live"


def test_stream_stop(client: TestClient):
    """Verifies pipeline stop endpoint."""
    response = client.post("/stream/stop")
    assert response.status_code == 200
    assert response.json()["status"] in ("idle", "stopped")


# ══════════════════════════════════════════════════════════════════════════════
#  Stream A: Server-Sent Events (SSE) Tests
# ══════════════════════════════════════════════════════════════════════════════
def test_stream_a_broadcaster_subscription():
    """Tests that the broadcaster can register queues and publish thread-safely."""
    loop = asyncio.new_event_loop()
    broadcaster.set_loop(loop)

    q = broadcaster.subscribe()
    assert q in broadcaster._subscribers

    # Publish dummy frame from simulated GStreamer thread
    test_event = {
        "frame_id": 1,
        "num_humans": 1,
        "pose_keypoints": [[[10.0, 20.0]]],
        "clip_embedding": [0.1] * 512,
    }
    broadcaster.publish_from_thread(test_event)

    # Run pending loop tasks
    loop.stop()
    loop.run_forever()

    assert not q.empty()
    item = q.get_nowait()
    assert item["frame_id"] == 1
    assert item["num_humans"] == 1

    broadcaster.unsubscribe(q)
    assert q not in broadcaster._subscribers
    loop.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Stream B: Protected Forensic Latent Endpoint Tests
# ══════════════════════════════════════════════════════════════════════════════
def test_stream_b_unauthorized_without_token(client: TestClient):
    """Rejects request without compliance token header with 401 Unauthorized."""
    response = client.get("/analytics/stream-b")
    assert response.status_code == 401
    assert "Missing compliance token" in response.json()["detail"]


def test_stream_b_forbidden_with_invalid_token(client: TestClient):
    """Rejects request with invalid or expired compliance token with 403 Forbidden."""
    mock_vault = MagicMock()
    mock_vault.verify_token.return_value = False
    pipeline_mgr.vault_manager = mock_vault

    headers = {"X-Vault-Token": "invalid-token-xyz"}
    response = client.get("/analytics/stream-b", headers=headers)
    assert response.status_code == 403
    assert "Access Denied" in response.json()["detail"]


def test_stream_b_authorized_with_drained_latents(client: TestClient):
    """Tests authorized extraction of encrypted face latents using atomic drain."""
    # Setup mocks
    mock_vault = MagicMock()
    mock_vault.verify_token.return_value = True
    pipeline_mgr.vault_manager = mock_vault

    mock_crypto = MagicMock()
    dummy_ckks_bytes = b"ENCRYPTED_CKKS_VECTOR_SERIALIZED_BYTES_128D"
    mock_crypto.encrypt_latent.return_value = dummy_ckks_bytes
    pipeline_mgr.crypto_engine = mock_crypto

    # Inject mock latent into AI engine store
    mock_ai = MagicMock()
    mock_store = LatentStore()
    mock_store.push([
        LatentRecord(
            frame_id=101,
            pts_ns=3000000000,
            mu=torch.zeros((1, 128), dtype=torch.float32),
            box=(50, 60, 150, 160),
        )
    ])
    mock_ai.latent_store = mock_store
    pipeline_mgr.ai_engine = mock_ai

    headers = {"X-Vault-Token": "valid-compliance-token-123"}
    response = client.get("/analytics/stream-b", headers=headers)
    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 1
    assert data["officer_token_verified"] is True
    latent = data["latents"][0]
    assert latent["frame_id"] == 101
    assert latent["box"] == [50, 60, 150, 160]
    assert base64.b64decode(latent["ckks_bytes_b64"]) == dummy_ckks_bytes

    # Ensure store was drained atomically (empty on subsequent call)
    assert len(mock_store.drain()) == 0


def test_stream_b_binary_format_response(client: TestClient):
    """Tests raw octet-stream binary response when format=binary is requested."""
    mock_vault = MagicMock()
    mock_vault.verify_token.return_value = True
    pipeline_mgr.vault_manager = mock_vault

    mock_crypto = MagicMock()
    dummy_ckks_bytes = b"RAW_BINARY_CKKS_PAYLOAD"
    mock_crypto.encrypt_latent.return_value = dummy_ckks_bytes
    pipeline_mgr.crypto_engine = mock_crypto

    mock_ai = MagicMock()
    mock_store = LatentStore()
    mock_store.push([
        LatentRecord(
            frame_id=102,
            pts_ns=3333333333,
            mu=torch.zeros((1, 128), dtype=torch.float32),
            box=(10, 20, 80, 90),
        )
    ])
    mock_ai.latent_store = mock_store
    pipeline_mgr.ai_engine = mock_ai

    headers = {"X-Compliance-Token": "valid-compliance-token-123"}
    response = client.get("/analytics/stream-b?format=binary", headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.content == dummy_ckks_bytes
