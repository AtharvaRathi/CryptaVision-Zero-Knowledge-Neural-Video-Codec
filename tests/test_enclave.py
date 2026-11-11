"""
CryptaVision Phase 7 — Unit & Integration Tests (Forensic Reconstruction Enclave)
================================================================================
Verifies:
  1. FaceVAE neural architecture, encode/decode tensor shapes, and range constraints.
  2. CiphertextLoader support for raw binary, Base64, Cloud JSON envelopes, and Phase 6 packets.
  3. VaultEnclaveAuthenticator authentication, SHA-256 integrity validation, and error traps.
  4. End-to-end CKKS homomorphic decryption -> FaceVAE decoding -> image export.
  5. Zero-trust memory hygiene: zeroing mutable buffers and forced garbage collection.
  6. Subprocess CLI invocation with output artifact verification.

Run with:
    pytest tests/test_enclave.py -v
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from crypto_engine import CryptoEngine, TENSEAL_AVAILABLE
from forensic_enclave.reconstructor import (
    CV2_AVAILABLE,
    PIL_AVAILABLE,
    CiphertextLoader,
    EnclaveAuthenticationError,
    EnclaveDecryptionError,
    EnclavePayloadError,
    EnclaveReconstructionError,
    FaceVAE,
    ForensicAuditRecord,
    ForensicReconstructionEnclave,
    VaultEnclaveAuthenticator,
)


# ══════════════════════════════════════════════════════════════════════════════
#  1. FaceVAE Neural Tests
# ══════════════════════════════════════════════════════════════════════════════
class TestFaceVAE:
    """Verifies FaceVAE structural shapes, latent dimensions, and decoder constraints."""

    @pytest.fixture
    def vae(self) -> FaceVAE:
        torch.manual_seed(42)
        return FaceVAE(latent_dim=128)

    def test_encode_decode_shapes(self, vae: FaceVAE):
        x = torch.rand(2, 3, 64, 64, dtype=torch.float32)
        mu, log_var = vae.encode(x)
        assert mu.shape == (2, 128)
        assert log_var.shape == (2, 128)

        recon = vae.decode(mu)
        assert recon.shape == (2, 3, 64, 64)
        # Sigmoid must enforce output in [0, 1]
        assert recon.min() >= 0.0
        assert recon.max() <= 1.0

    def test_forward_pass(self, vae: FaceVAE):
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        recon, mu, log_var = vae(x)
        assert recon.shape == (1, 3, 64, 64)
        assert mu.shape == (1, 128)
        assert log_var.shape == (1, 128)


# ══════════════════════════════════════════════════════════════════════════════
#  2. CiphertextLoader Tests
# ══════════════════════════════════════════════════════════════════════════════
class TestCiphertextLoader:
    """Tests multi-format ciphertext ingestion and packet parsing."""

    def test_load_raw_binary(self, tmp_path: Path):
        raw_bytes = b"MOCK_RAW_CIPHERTEXT_BINARY_DATA_1234567890" * 4
        file_path = tmp_path / "target.bin"
        file_path.write_bytes(raw_bytes)

        loaded_bytes, digest = CiphertextLoader.load_bytes_from_file(file_path)
        assert loaded_bytes == raw_bytes
        assert digest == hashlib.sha256(raw_bytes).hexdigest()

    def test_load_base64_text(self, tmp_path: Path):
        raw_bytes = b"MOCK_BASE64_CIPHERTEXT_DATA_PAYLOAD_998877" * 3
        b64_str = base64.b64encode(raw_bytes).decode("ascii")
        file_path = tmp_path / "target.b64"
        file_path.write_text(b64_str)

        loaded_bytes, digest = CiphertextLoader.load_bytes_from_file(file_path)
        assert loaded_bytes == raw_bytes
        assert digest == hashlib.sha256(raw_bytes).hexdigest()

    def test_load_json_envelope_stream_b(self, tmp_path: Path):
        raw_face_0 = b"FACE_0_ENCRYPTED_CKKS_PAYLOAD" * 4
        raw_face_1 = b"FACE_1_ENCRYPTED_CKKS_PAYLOAD" * 4

        payload = {
            "status": "success",
            "count": 2,
            "latents": [
                {
                    "frame_id": 100,
                    "face_idx": 0,
                    "box": [10, 20, 50, 60],
                    "ckks_bytes_b64": base64.b64encode(raw_face_0).decode("ascii"),
                },
                {
                    "frame_id": 100,
                    "face_idx": 1,
                    "box": [120, 140, 60, 70],
                    "ckks_bytes_b64": base64.b64encode(raw_face_1).decode("ascii"),
                },
            ],
        }
        file_path = tmp_path / "forensic_export.json"
        file_path.write_text(json.dumps(payload))

        # Default face_index = 0
        loaded_0, digest_0 = CiphertextLoader.load_bytes_from_file(file_path, face_index=0)
        assert loaded_0 == raw_face_0
        assert digest_0 == hashlib.sha256(raw_face_0).hexdigest()

        # Face index = 1
        loaded_1, digest_1 = CiphertextLoader.load_bytes_from_file(file_path, face_index=1)
        assert loaded_1 == raw_face_1
        assert digest_1 == hashlib.sha256(raw_face_1).hexdigest()

    def test_load_phase6_binary_packet(self, tmp_path: Path):
        ckks_data = b"PHASE_6_EDGE_CKKS_PAYLOAD_BINARY_DATA" * 5
        env_meta = json.dumps({"frame_id": 42}).encode("utf-8")
        env_len = len(env_meta)

        packet_body = struct.pack("!I", env_len) + env_meta + ckks_data
        packet = struct.pack("!II", 0x43525950, len(packet_body)) + packet_body

        file_path = tmp_path / "packet.raw"
        file_path.write_bytes(packet)

        loaded, digest = CiphertextLoader.load_bytes_from_file(file_path)
        assert loaded == ckks_data
        assert digest == hashlib.sha256(ckks_data).hexdigest()

    def test_missing_file_raises_payload_error(self, tmp_path: Path):
        with pytest.raises(EnclavePayloadError, match="not found"):
            CiphertextLoader.load_bytes_from_file(tmp_path / "non_existent.bin")

    def test_empty_file_raises_payload_error(self, tmp_path: Path):
        empty_file = tmp_path / "empty.bin"
        empty_file.write_bytes(b"")
        with pytest.raises(EnclavePayloadError, match="empty"):
            CiphertextLoader.load_bytes_from_file(empty_file)

    def test_face_index_out_of_bounds(self, tmp_path: Path):
        payload = {"latents": [{"ckks_bytes_b64": base64.b64encode(b"DATA").decode("ascii")}]}
        file_path = tmp_path / "single_face.json"
        file_path.write_text(json.dumps(payload))

        with pytest.raises(EnclavePayloadError, match="exceeds available"):
            CiphertextLoader.load_bytes_from_file(file_path, face_index=5)


# ══════════════════════════════════════════════════════════════════════════════
#  3. VaultEnclaveAuthenticator Tests
# ══════════════════════════════════════════════════════════════════════════════
class TestVaultEnclaveAuthenticator:
    """Verifies Vault token auth and secret context integrity verification."""

    @patch("forensic_enclave.reconstructor.hvac.Client")
    def test_successful_vault_retrieval(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_client_cls.return_value = mock_client

        raw_secret_context = b"SECRET_CONTEXT_PRIVATE_KEY_BYTES_9999"
        b64_payload = base64.b64encode(raw_secret_context).decode("ascii")
        sha256_digest = hashlib.sha256(raw_secret_context).hexdigest()

        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {
                "data": {
                    "secret_context_b64": b64_payload,
                    "sha256_checksum": sha256_digest,
                }
            }
        }

        auth = VaultEnclaveAuthenticator(
            vault_url="http://vault.internal:8200",
            vault_token="s.COMPLIANCE_TOKEN_AUTHORIZED_123",
            mount_point="secret",
            secret_path="cryptavision/keys/tenseal_ckks_secret_context",
        )

        retrieved = auth.retrieve_secret_context()
        assert retrieved == raw_secret_context

    @patch("forensic_enclave.reconstructor.hvac.Client")
    def test_tamper_detection_checksum_mismatch(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_client_cls.return_value = mock_client

        raw_secret_context = b"TAMPERED_SECRET_CONTEXT_BYTES"
        b64_payload = base64.b64encode(raw_secret_context).decode("ascii")

        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {
                "data": {
                    "secret_context_b64": b64_payload,
                    "sha256_checksum": "0000000000000000000000000000000000000000000000000000000000000000",
                }
            }
        }

        auth = VaultEnclaveAuthenticator(
            vault_url="http://vault.internal:8200",
            vault_token="s.COMPLIANCE_TOKEN_AUTHORIZED_123",
        )

        with pytest.raises(EnclaveAuthenticationError, match="Cryptographic digest mismatch"):
            auth.retrieve_secret_context()

    @patch("forensic_enclave.reconstructor.hvac.Client")
    def test_unauthenticated_token_raises_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = False
        mock_client_cls.return_value = mock_client

        auth = VaultEnclaveAuthenticator(
            vault_url="http://vault.internal:8200",
            vault_token="INVALID_EXPIRED_TOKEN",
        )

        with pytest.raises(EnclaveAuthenticationError, match="failed authentication"):
            auth.connect_and_authenticate()


# ══════════════════════════════════════════════════════════════════════════════
#  4. End-to-End TenSEAL CKKS Decryption & Reconstruction
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not TENSEAL_AVAILABLE, reason="TenSEAL library not installed")
class TestForensicReconstructionEndToEnd:
    """Verifies real cryptographic decryption, VAE decode, and memory hygiene."""

    @pytest.fixture
    def crypto_setup(self) -> Tuple[CryptoEngine, bytes, torch.Tensor]:
        """Creates CryptoEngine, serializes secret context, and encrypts a test latent."""
        engine = CryptoEngine(latent_dim=128)
        secret_context_bytes = engine.serialize_secret_context()

        torch.manual_seed(1337)
        original_mu = torch.randn(1, 128, dtype=torch.float32)
        ciphertext = engine.encrypt_latent(original_mu)

        return engine, secret_context_bytes, original_mu

    def test_full_reconstruction_pipeline(
        self, crypto_setup: Tuple[CryptoEngine, bytes, torch.Tensor], tmp_path: Path
    ):
        engine, secret_context_bytes, original_mu = crypto_setup
        ciphertext = engine.encrypt_latent(original_mu)

        output_image = tmp_path / "suspect_reconstructed.png"

        enclave = ForensicReconstructionEnclave(
            secret_context_bytes=secret_context_bytes,
            latent_dim=128,
            device="cpu",
        )

        # Execute full reconstruction
        img_rgb, out_sha256 = enclave.reconstruct(
            ciphertext_bytes=ciphertext,
            output_path=output_image,
            scrub_memory=True,
        )

        # Validate image dimensions & values
        assert output_image.is_file()
        assert output_image.stat().st_size > 0
        assert img_rgb.shape == (64, 64, 3)
        assert img_rgb.dtype == np.uint8
        assert len(out_sha256) == 64

    def test_decryption_numerical_fidelity(
        self, crypto_setup: Tuple[CryptoEngine, bytes, torch.Tensor]
    ):
        engine, secret_context_bytes, original_mu = crypto_setup
        ciphertext = engine.encrypt_latent(original_mu)

        enclave = ForensicReconstructionEnclave(
            secret_context_bytes=secret_context_bytes,
            latent_dim=128,
            device="cpu",
        )

        z_decrypted = enclave.decrypt_vector(ciphertext)
        assert z_decrypted.shape == (1, 128)

        # CKKS precision check: max absolute error < 1e-4
        max_error = torch.max(torch.abs(original_mu - z_decrypted)).item()
        assert max_error < 1e-4, f"CKKS decryption precision too low: {max_error}"

    def test_memory_scrubbing(self):
        """Verifies zero-trust tensor zeroing and memory scrubbing."""
        tensor = torch.ones(1, 128, dtype=torch.float32)
        raw_list = [1.0, 2.0, 3.0]
        raw_buffer = bytearray(b"SENSITIVE_KEY_BYTES")

        ForensicReconstructionEnclave.scrub_memory(tensor, raw_list, raw_buffer)

        assert torch.all(tensor == 0.0)
        assert all(x == 0.0 for x in raw_list)
        assert raw_buffer == b"\x00" * len(raw_buffer)


# ══════════════════════════════════════════════════════════════════════════════
#  5. Subprocess CLI Invocation Tests
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not TENSEAL_AVAILABLE, reason="TenSEAL library not installed")
class TestForensicEnclaveCLI:
    """Tests the forensic CLI end-to-end using file-based air-gapped credentials."""

    def test_cli_execution_with_json_ciphertext(self, tmp_path: Path):
        engine = CryptoEngine(latent_dim=128)
        secret_context = engine.serialize_secret_context()

        # Save secret key file
        key_file = tmp_path / "secret_context.bin"
        key_file.write_bytes(secret_context)

        # Encrypt a sample latent
        test_mu = torch.randn(1, 128, dtype=torch.float32)
        cipher_bytes = engine.encrypt_latent(test_mu)

        # Save as JSON envelope (mimicking GET /analytics/stream-b)
        json_file = tmp_path / "stream_b_export.json"
        json_file.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "latents": [
                        {
                            "frame_id": 1,
                            "face_idx": 0,
                            "box": [50, 50, 100, 100],
                            "ckks_bytes_b64": base64.b64encode(cipher_bytes).decode("ascii"),
                        }
                    ],
                }
            )
        )

        out_img = tmp_path / "suspect_reconstructed.png"
        audit_log = tmp_path / "audit.json"

        # Execute reconstructor CLI
        cmd = [
            sys.executable,
            str(REPO_ROOT / "forensic_enclave" / "reconstructor.py"),
            "--ciphertext",
            str(json_file),
            "--secret-key-file",
            str(key_file),
            "--output",
            str(out_img),
            "--audit-log",
            str(audit_log),
            "--incident-id",
            "INC-TEST-404",
            "--officer-id",
            "OFFICER-TEST-99",
        ]

        import subprocess

        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"CLI failed with error:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"

        # Check output artifacts
        assert out_img.is_file()
        assert out_img.stat().st_size > 0
        assert audit_log.is_file()

        audit_data = json.loads(audit_log.read_text())
        assert audit_data["incident_id"] == "INC-TEST-404"
        assert audit_data["officer_id"] == "OFFICER-TEST-99"
        assert audit_data["memory_scrubbed"] is True
