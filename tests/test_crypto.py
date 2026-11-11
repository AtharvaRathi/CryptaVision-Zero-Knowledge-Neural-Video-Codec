"""
CryptaVision Phase 3 — Unit & Integration Tests (HE & Key Management)
=====================================================================
Comprehensive test suite verifying:
1. CKKS Homomorphic Encryption parameter rigor (N=8192, [60,40,40,60], scale=2^40).
2. Encryption and decryption fidelity for [1, 128] FP32 PyTorch tensors.
3. Secret vs. Public context separation and unauthorized decryption prevention.
4. HashiCorp Vault KV v2 storage with SHA-256 cryptographic integrity validation.
5. Ephemeral compliance officer token issuance (1-hour TTL, non-renewable, audit metadata).
6. Resilient fallback to in-memory KV v2 storage when Vault is offline.

Run with:
    pytest tests/test_crypto.py -v
"""

from __future__ import annotations

import base64
import hashlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from crypto_engine import CryptoEngine, TENSEAL_AVAILABLE
from vault_manager import (
    ComplianceTokenRequest,
    ComplianceTokenResponse,
    IntegrityVerificationError,
    VaultConfig,
    VaultConnectionError,
    VaultManager,
)


# ══════════════════════════════════════════════════════════════════════════════
#  CryptoEngine Tests (Homomorphic Encryption)
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not TENSEAL_AVAILABLE, reason="TenSEAL library not installed")
class TestCryptoEngine:
    """Tests for TenSEAL CKKS Homomorphic Encryption Engine."""

    @pytest.fixture
    def engine(self) -> CryptoEngine:
        """Instantiates a fresh CryptoEngine with standard parameters."""
        return CryptoEngine(latent_dim=128)

    @pytest.fixture
    def sample_latent(self) -> torch.Tensor:
        """Generates a realistic normalized [1, 128] face latent embedding."""
        torch.manual_seed(42)
        raw = torch.randn(1, 128, dtype=torch.float32)
        # Normalize to unit sphere (typical for face recognition embeddings)
        return raw / torch.norm(raw, p=2, dim=-1, keepdim=True)

    def test_ckks_parameters(self, engine: CryptoEngine):
        """Verifies exact adherence to cryptographic directives."""
        assert engine.POLY_MODULUS_DEGREE == 8192
        assert engine.COEFF_MOD_BIT_SIZES == [60, 40, 40, 60]
        assert engine.SCALE == float(2**40)
        assert engine.latent_dim == 128
        assert engine.is_private is True

    def test_encrypt_decrypt_roundtrip(self, engine: CryptoEngine, sample_latent: torch.Tensor):
        """Tests that encryption and decryption preserve tensor shape and high numerical fidelity."""
        # Encrypt
        encrypted_bytes = engine.encrypt_latent(sample_latent)
        assert isinstance(encrypted_bytes, bytes)
        assert len(encrypted_bytes) > 0

        # Decrypt
        decrypted_tensor = engine.decrypt_latent(encrypted_bytes)

        # Assert shape and dtype
        assert decrypted_tensor.shape == (1, 128)
        assert decrypted_tensor.dtype == torch.float32

        # Assert mathematical precision (CKKS noise floor < 1e-4)
        metrics = engine.compute_reconstruction_metrics(sample_latent, decrypted_tensor)
        assert metrics["max_abs_error"] < 1e-4, f"Max error too high: {metrics['max_abs_error']}"
        assert metrics["mean_abs_error"] < 1e-5
        assert metrics["snr_db"] > 60.0  # Signal-to-Noise Ratio > 60 dB

    def test_encrypt_validates_dimensions(self, engine: CryptoEngine):
        """Rejects tensors that do not have 128 elements."""
        wrong_size = torch.randn(1, 64)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            engine.encrypt_latent(wrong_size)

        wrong_size_2d = torch.randn(2, 128)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            engine.encrypt_latent(wrong_size_2d)

    def test_encrypt_handles_grad_and_cuda_tensors(self, engine: CryptoEngine):
        """Verifies tensor with gradients is safely detached without error."""
        tensor_with_grad = torch.randn(1, 128, requires_grad=True)
        encrypted_bytes = engine.encrypt_latent(tensor_with_grad)
        assert isinstance(encrypted_bytes, bytes)

    def test_public_context_cannot_decrypt(self, engine: CryptoEngine, sample_latent: torch.Tensor):
        """Confirms that a public evaluation context cannot decrypt (zero-trust worker node)."""
        encrypted_bytes = engine.encrypt_latent(sample_latent)

        # Export public context only (no secret key)
        public_context_bytes = engine.serialize_public_context()
        public_engine = CryptoEngine.from_serialized_context(public_context_bytes, latent_dim=128)

        assert public_engine.is_private is False

        # Attempting decryption must raise PermissionError
        with pytest.raises(PermissionError, match="does not contain the secret key"):
            public_engine.decrypt_latent(encrypted_bytes)

    def test_context_serialization_roundtrip(self, engine: CryptoEngine, sample_latent: torch.Tensor):
        """Verifies full secret context serialization and restoration."""
        encrypted_bytes = engine.encrypt_latent(sample_latent)

        # Export secret context
        secret_context_bytes = engine.serialize_secret_context()

        # Restore engine from bytes
        restored_engine = CryptoEngine.from_serialized_context(secret_context_bytes, latent_dim=128)
        assert restored_engine.is_private is True

        decrypted = restored_engine.decrypt_latent(encrypted_bytes)
        assert decrypted.shape == (1, 128)
        assert torch.allclose(sample_latent, decrypted, atol=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
#  VaultManager Tests (Key Management & Access Control)
# ══════════════════════════════════════════════════════════════════════════════
class TestVaultManager:
    """Tests for HashiCorp Vault Key & Token Management."""

    @pytest.fixture
    def manager(self) -> VaultManager:
        """Instantiates VaultManager with in-memory fallback enabled."""
        config = VaultConfig(
            vault_url="http://127.0.0.1:9999",  # unreachable test address
            fallback_to_memory=True,
            timeout_seconds=1,
        )
        return VaultManager(config)

    def test_fallback_activation(self, manager: VaultManager):
        """Ensures fallback mode engages gracefully when Vault is unreachable."""
        assert manager.is_fallback is True
        assert manager.is_connected is False

    def test_strict_mode_raises_connection_error(self):
        """Ensures VaultConnectionError is raised when fallback_to_memory is disabled."""
        config = VaultConfig(
            vault_url="http://127.0.0.1:9999",
            fallback_to_memory=False,
            timeout_seconds=1,
        )
        with pytest.raises(VaultConnectionError):
            VaultManager(config)

    def test_store_and_retrieve_secret_context(self, manager: VaultManager):
        """Validates storage and retrieval of secret context bytes with integrity checks."""
        dummy_context = b"CRYPTOVISION_SECRET_CONTEXT_MOCK_BYTES_12345"
        
        # Store
        res = manager.store_secret_context(dummy_context, key_name="test_context")
        assert res["version"] == 1
        assert res["sha256"] == hashlib.sha256(dummy_context).hexdigest()

        # Retrieve
        retrieved = manager.retrieve_secret_context(key_name="test_context")
        assert retrieved == dummy_context

    def test_tamper_detection(self, manager: VaultManager):
        """Detects tampering if SHA-256 digest fails to match stored payload."""
        dummy_context = b"LEGITIMATE_CONTEXT_DATA"
        manager.store_secret_context(dummy_context, key_name="tamper_target")

        # Corrupt data in memory
        path = f"{manager.config.key_path}/tamper_target"
        corrupted = b"TAMPERED_MALICIOUS_CONTEXT"
        manager._memory_storage._secrets[path][-1]["data"]["secret_context_b64"] = (
            base64.b64encode(corrupted).decode("ascii")
        )

        # Retrieval must fail with IntegrityVerificationError
        with pytest.raises(IntegrityVerificationError):
            manager.retrieve_secret_context(key_name="tamper_target")

    def test_store_and_retrieve_encrypted_latent(self, manager: VaultManager):
        """Tests saving and retrieving encrypted face latent ciphertexts."""
        dummy_ciphertext = os.urandom(256)
        path = manager.store_encrypted_latent(
            frame_id=42,
            face_idx=0,
            encrypted_bytes=dummy_ciphertext,
            metadata={"confidence": 0.98},
        )
        assert "frame_42_face_0" in path

        retrieved = manager.retrieve_encrypted_latent(path)
        assert retrieved == dummy_ciphertext

    def test_generate_compliance_token(self, manager: VaultManager):
        """Validates generation of time-limited compliance officer investigation token."""
        req = ComplianceTokenRequest(
            officer_id="OFFICER-7729",
            incident_id="INC-2026-X99",
            ttl="1h",
            reason="Legal warrant compliance audit",
            num_uses=5,
        )
        resp = manager.generate_compliance_token(req)

        assert isinstance(resp, ComplianceTokenResponse)
        assert resp.lease_duration_seconds == 3600
        assert resp.meta["officer_id"] == "OFFICER-7729"
        assert resp.meta["incident_id"] == "INC-2026-X99"
        assert "cryptavision-compliance-officer" in resp.policies

        # Verify token validation
        assert manager.verify_token(resp.client_token) is True

        # Verify token revocation
        assert manager.revoke_token(resp.client_token) is True
        assert manager.verify_token(resp.client_token) is False

    @patch("vault_manager.hvac.Client")
    def test_live_vault_client_interaction(self, mock_hvac_client_cls):
        """Verifies proper hvac API calls when connected to a live Vault."""
        mock_client = MagicMock()
        mock_client.sys.is_initialized.return_value = True
        mock_client.is_authenticated.return_value = True
        mock_client.secrets.kv.v2.create_or_update_secret.return_value = {"data": {"version": 2}}
        mock_client.auth.token.create.return_value = {
            "auth": {
                "client_token": "s.liveVaultToken123",
                "accessor": "acc.123",
                "lease_duration": 3600,
            }
        }
        mock_hvac_client_cls.return_value = mock_client

        config = VaultConfig(
            vault_url="http://vault.internal:8200",
            vault_token="s.rootToken",
            fallback_to_memory=False,
        )
        mgr = VaultManager(config)

        assert mgr.is_connected is True
        assert mgr.is_fallback is False

        # Store context via mocked Vault
        res = mgr.store_secret_context(b"mock_key", key_name="vault_key")
        assert res["version"] == 2
        mock_client.secrets.kv.v2.create_or_update_secret.assert_called_once()

        # Generate compliance token via mocked Vault
        req = ComplianceTokenRequest(
            officer_id="OFF-01",
            incident_id="INC-01",
            ttl="1h",
        )
        tok_res = mgr.generate_compliance_token(req)
        assert tok_res.client_token == "s.liveVaultToken123"
        assert tok_res.is_fallback is False
        mock_client.auth.token.create.assert_called_once()
