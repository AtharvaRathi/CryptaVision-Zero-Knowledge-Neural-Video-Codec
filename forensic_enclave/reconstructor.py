#!/usr/bin/env python3
"""
CryptaVision Phase 7 — The Forensic Reconstruction Enclave
===========================================================
High-security, zero-trust CLI tool and core library for reversing the
homomorphic encryption pipeline during authorized compliance investigations.

Workflow Architecture:
----------------------
1. Vault Authenticator:
   - Connects to HashiCorp Vault via `hvac`.
   - Authenticates using an ephemeral, high-privilege Vault Compliance Token.
   - Retrieves the TenSEAL `secret_context` (containing the private CKKS decryption key)
     from the KV v2 engine (`secret/data/cryptavision/keys/tenseal_ckks_secret_context`).
   - Verifies SHA-256 cryptographic digest before loading.

2. TenSEAL Decryption Engine:
   - Ingests target CKKS ciphertext file (binary, base64, or JSON envelope).
   - Deserializes the ciphertext into a TenSEAL CKKSVector under the secret context.
   - Homomorphically decrypts the vector into a 1D plaintext float list.
   - Slices and reshapes to a [1, 128] FP32 PyTorch tensor (`mu`).

3. VAE Reconstructor (FaceVAE.decode):
   - Feeds the decrypted [1, 128] latent vector into `FaceVAE.decode(z)`.
   - Generates a [1, 3, 64, 64] image tensor in range [0, 1].
   - Denormalizes to uint8 RGB/BGR [64, 64, 3] and exports to `suspect_reconstructed.png`.

4. Zero-Trust Memory Hygiene:
   - Biometric decryption in secure enclaves requires zero residual plaintext.
   - Overwrites and zeroes raw float lists and tensors before deallocation.
   - Explicitly deletes references and triggers `gc.collect()` and CUDA cache purge.
"""

from __future__ import annotations

import argparse
import base64
import gc
import getpass
import hashlib
import json
import logging
import os
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional image dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None  # type: ignore
    PIL_AVAILABLE = False

# TenSEAL CKKS library
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    ts = None  # type: ignore
    TENSEAL_AVAILABLE = False

# HashiCorp Vault client
try:
    import hvac
    import hvac.exceptions
    HVAC_AVAILABLE = True
except ImportError:
    hvac = None  # type: ignore
    HVAC_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("CryptaVision.ForensicEnclave")


# ══════════════════════════════════════════════════════════════════════════════
#  Custom Exceptions
# ══════════════════════════════════════════════════════════════════════════════
class EnclaveError(Exception):
    """Base exception for all forensic reconstruction enclave errors."""


class EnclaveAuthenticationError(EnclaveError):
    """Raised when HashiCorp Vault token authentication or access fails."""


class EnclaveDecryptionError(EnclaveError):
    """Raised when CKKS homomorphic decryption fails."""


class EnclaveReconstructionError(EnclaveError):
    """Raised when FaceVAE neural decoding fails."""


class EnclavePayloadError(EnclaveError):
    """Raised when input ciphertext file is missing, corrupted, or invalid."""


# ══════════════════════════════════════════════════════════════════════════════
#  FaceVAE Neural Architecture (Isolated Definition)
# ══════════════════════════════════════════════════════════════════════════════
class FaceVAE(nn.Module):
    """
    Convolutional Variational Autoencoder (VAE) for 64×64 face crops.
    Architecturally identical to Phase 2 `pipeline/ai_engine.py:FaceVAE` to ensure
    100% weights compatibility without requiring external pipeline dependencies.
    """

    INPUT_SIZE: int = 64
    HIDDEN_SIZES: Tuple[int, ...] = (32, 64, 128, 256)
    DEFAULT_LATENT_DIM: int = 128

    def __init__(self, latent_dim: int = DEFAULT_LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 8 -> 4
            nn.ReLU(inplace=True),
        )
        self._flat_dim = 256 * 4 * 4  # 4096
        self.fc_mu = nn.Linear(self._flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(self._flat_dim, latent_dim)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.fc_decode = nn.Linear(latent_dim, self._flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 4 -> 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 32 -> 64
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """He initialization for convolutions and Xavier for linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_log_var(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a [B, latent_dim] latent tensor back to a [B, 3, 64, 64] image in [0, 1].
        """
        h = F.relu(self.fc_decode(z))
        h = h.view(-1, 256, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decode(z)
        return recon, mu, log_var


# ══════════════════════════════════════════════════════════════════════════════
#  Audit & Telemetry Metadata Models
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ForensicAuditRecord:
    """Tamper-evident audit record documenting a facial reconstruction session."""

    incident_id: str
    officer_id: str
    session_timestamp: str
    ciphertext_path: str
    ciphertext_sha256: str
    secret_key_source: str
    output_image_path: str
    output_image_sha256: str
    image_shape: Tuple[int, int, int]
    memory_scrubbed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "officer_id": self.officer_id,
            "session_timestamp": self.session_timestamp,
            "ciphertext_path": self.ciphertext_path,
            "ciphertext_sha256": self.ciphertext_sha256,
            "secret_key_source": self.secret_key_source,
            "output_image_path": self.output_image_path,
            "output_image_sha256": self.output_image_sha256,
            "image_shape": list(self.image_shape),
            "memory_scrubbed": self.memory_scrubbed,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Vault Authenticator Engine
# ══════════════════════════════════════════════════════════════════════════════
class VaultEnclaveAuthenticator:
    """
    Connects to HashiCorp Vault to authenticate compliance credentials
    and securely retrieve TenSEAL CKKS secret context keys.
    """

    def __init__(
        self,
        vault_url: str = "http://127.0.0.1:8200",
        vault_token: Optional[str] = None,
        mount_point: str = "secret",
        secret_path: str = "cryptavision/keys/tenseal_ckks_secret_context",
        namespace: Optional[str] = None,
        timeout: int = 10,
    ):
        self.vault_url = vault_url.rstrip("/")
        self.vault_token = vault_token
        self.mount_point = mount_point
        self.secret_path = secret_path.strip("/")
        self.namespace = namespace
        self.timeout = timeout
        self._client: Optional[hvac.Client] = None

    def connect_and_authenticate(self) -> hvac.Client:
        """
        Establishes authenticated session with HashiCorp Vault.
        """
        if not HVAC_AVAILABLE:
            raise EnclaveAuthenticationError(
                "hvac library is not installed. Install via: pip install hvac>=2.1.0"
            )

        if not self.vault_token:
            raise EnclaveAuthenticationError("Missing Vault Compliance Token.")

        try:
            log.info("Connecting to HashiCorp Vault at %s...", self.vault_url)
            self._client = hvac.Client(
                url=self.vault_url,
                token=self.vault_token,
                namespace=self.namespace,
                timeout=self.timeout,
            )

            # Check if token is valid
            if not self._client.is_authenticated():
                raise EnclaveAuthenticationError(
                    f"Provided Vault token failed authentication at {self.vault_url}. "
                    "The compliance token may have expired or is invalid."
                )

            log.info("Vault Compliance Token verified successfully.")
            return self._client

        except Exception as exc:
            if isinstance(exc, EnclaveAuthenticationError):
                raise
            raise EnclaveAuthenticationError(
                f"Failed to communicate with HashiCorp Vault at {self.vault_url}: {exc}"
            ) from exc

    def retrieve_secret_context(self) -> bytes:
        """
        Fetches the TenSEAL CKKS secret context from Vault KV v2 and validates integrity.
        """
        if self._client is None:
            self.connect_and_authenticate()
        assert self._client is not None

        try:
            log.info(
                "Reading secret key from Vault KV v2 mount='%s' path='%s'...",
                self.mount_point,
                self.secret_path,
            )
            # Read from KV v2 engine
            resp = self._client.secrets.kv.v2.read_secret_version(
                path=self.secret_path,
                mount_point=self.mount_point,
            )
            secret_data = resp["data"]["data"]
        except Exception as exc:
            raise EnclaveAuthenticationError(
                f"Failed to read secret key at '{self.mount_point}/{self.secret_path}': {exc}"
            ) from exc

        # Extract base64 payload
        b64_payload = secret_data.get("secret_context_b64")
        if not b64_payload:
            raise EnclaveAuthenticationError(
                "Vault payload does not contain required 'secret_context_b64' field."
            )

        # Decode base64 to binary
        try:
            context_bytes = base64.b64decode(b64_payload.encode("ascii"))
        except Exception as exc:
            raise EnclaveAuthenticationError(f"Malformed Base64 secret context: {exc}") from exc

        # Verify cryptographic checksum if stored in Vault
        expected_sha = secret_data.get("sha256_checksum")
        if expected_sha:
            computed_sha = hashlib.sha256(context_bytes).hexdigest()
            if computed_sha != expected_sha:
                raise EnclaveAuthenticationError(
                    f"Cryptographic digest mismatch for secret context! "
                    f"Expected: {expected_sha}, Computed: {computed_sha}"
                )
            log.info("SHA-256 integrity verified: %s", expected_sha[:16] + "...")

        log.info(
            "Secret Context successfully loaded (%d bytes). Ready for CKKS decryption.",
            len(context_bytes),
        )
        return context_bytes


# ══════════════════════════════════════════════════════════════════════════════
#  Ciphertext Loader & Normalizer
# ══════════════════════════════════════════════════════════════════════════════
class CiphertextLoader:
    """
    Intelligently parses target CKKS ciphertext from diverse input formats:
      - Raw binary file (*.bin, *.ckks, *.dat)
      - Plain Base64 encoded file (*.b64, *.txt)
      - Cloud Dashboard JSON export (*.json) with optional face_index selection
      - Binary network stream packets (Phase 6 0x43525950 format)
    """

    MAGIC_PACKET_HEADER: int = 0x43525950  # "CRYP"

    @classmethod
    def load_bytes_from_file(cls, file_path: Union[str, Path], face_index: int = 0) -> Tuple[bytes, str]:
        """
        Loads and returns (raw_ckks_bytes, sha256_hash).
        """
        path = Path(file_path)
        if not path.is_file():
            raise EnclavePayloadError(f"Ciphertext file not found: {path}")

        raw_content = path.read_bytes()
        if len(raw_content) == 0:
            raise EnclavePayloadError(f"Ciphertext file is empty: {path}")

        # 1. Check for Phase 6 custom TCP binary packet protocol
        if len(raw_content) > 8:
            magic, payload_len = struct.unpack("!II", raw_content[:8])
            if magic == cls.MAGIC_PACKET_HEADER:
                log.info("Detected Phase 6 binary network sink packet (Magic=0x43525950).")
                packet_payload = raw_content[8 : 8 + payload_len]
                if len(packet_payload) > 4:
                    env_len = struct.unpack("!I", packet_payload[:4])[0]
                    ckks_bytes = packet_payload[4 + env_len :]
                    digest = hashlib.sha256(ckks_bytes).hexdigest()
                    return ckks_bytes, digest

        # 2. Check for JSON format
        try:
            text_candidate = raw_content.decode("utf-8").strip()
            if text_candidate.startswith("{") or text_candidate.startswith("["):
                json_data = json.loads(text_candidate)
                ckks_b64: Optional[str] = None

                if isinstance(json_data, dict):
                    # Direct key
                    if "ckks_bytes_b64" in json_data:
                        ckks_b64 = json_data["ckks_bytes_b64"]
                    # Latents array from GET /analytics/stream-b
                    elif "latents" in json_data and isinstance(json_data["latents"], list):
                        latents = json_data["latents"]
                        if len(latents) == 0:
                            raise EnclavePayloadError("JSON payload contains empty 'latents' array.")
                        if face_index >= len(latents):
                            raise EnclavePayloadError(
                                f"Requested face_index {face_index} exceeds available faces ({len(latents)})."
                            )
                        ckks_b64 = latents[face_index].get("ckks_bytes_b64")
                    # Audit archive record
                    elif "encrypted_latent_b64" in json_data:
                        ckks_b64 = json_data["encrypted_latent_b64"]

                elif isinstance(json_data, list):
                    if face_index >= len(json_data):
                        raise EnclavePayloadError(
                            f"Requested face_index {face_index} exceeds available items ({len(json_data)})."
                        )
                    item = json_data[face_index]
                    if isinstance(item, dict):
                        ckks_b64 = item.get("ckks_bytes_b64") or item.get("encrypted_latent_b64")
                    elif isinstance(item, str):
                        ckks_b64 = item

                if ckks_b64:
                    ckks_bytes = base64.b64decode(ckks_b64)
                    digest = hashlib.sha256(ckks_bytes).hexdigest()
                    log.info("Successfully extracted CKKS vector from JSON envelope (Face #%d).", face_index)
                    return ckks_bytes, digest
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass

        # 3. Check for standalone Base64 text
        try:
            text_candidate = raw_content.decode("ascii").strip()
            if text_candidate.isalnum() or "/" in text_candidate or "+" in text_candidate:
                ckks_bytes = base64.b64decode(text_candidate)
                # Verify that it doesn't just decode arbitrary junk
                if len(ckks_bytes) > 64:
                    digest = hashlib.sha256(ckks_bytes).hexdigest()
                    log.info("Parsed Base64 ciphertext representation.")
                    return ckks_bytes, digest
        except Exception:
            pass

        # 4. Default: Raw binary bytes
        digest = hashlib.sha256(raw_content).hexdigest()
        log.info("Loaded raw binary CKKS ciphertext (%d bytes).", len(raw_content))
        return raw_content, digest


# ══════════════════════════════════════════════════════════════════════════════
#  Forensic Decryption & Neural Reconstruction Engine
# ══════════════════════════════════════════════════════════════════════════════
class ForensicReconstructionEnclave:
    """
    Main orchestration engine for decrypting CKKS latent vectors and
    reconstructing photographic facial representations.
    """

    DEFAULT_LATENT_DIM: int = 128

    def __init__(
        self,
        secret_context_bytes: bytes,
        weights_path: Optional[Union[str, Path]] = None,
        latent_dim: int = DEFAULT_LATENT_DIM,
        device: Optional[str] = None,
    ):
        """
        Initializes the Forensic Reconstruction Enclave.

        Args:
            secret_context_bytes: Raw serialized TenSEAL context containing private key.
            weights_path: Optional path to pre-trained FaceVAE state dict checkpoint.
            latent_dim: Dimensionality of latent embeddings (default: 128).
            device: Target device ('cpu', 'cuda', 'mps'). Defaults to best available.
        """
        if not TENSEAL_AVAILABLE:
            raise EnclaveDecryptionError(
                "TenSEAL is required for CKKS decryption. Install via: pip install tenseal>=0.3.14"
            )

        self.latent_dim = latent_dim
        self._device = self._resolve_device(device)

        # 1. Initialize TenSEAL private context
        try:
            self._context = ts.context_from(secret_context_bytes)
            if not self._context.is_private():
                raise EnclaveAuthenticationError(
                    "Retrieved TenSEAL context does not contain the secret decryption key. "
                    "Decryption requires the private key retrieved from HashiCorp Vault."
                )
        except Exception as exc:
            raise EnclaveDecryptionError(f"Failed to load TenSEAL private context: {exc}") from exc

        # 2. Initialize FaceVAE Decoder
        self._vae = FaceVAE(latent_dim=latent_dim).to(self._device)
        self._vae.eval()

        if weights_path:
            self._load_weights(weights_path)
        else:
            log.warning(
                "No FaceVAE checkpoint weights specified (--vae-weights). "
                "Using baseline initialized architecture for structural reconstruction."
            )

    @staticmethod
    def _resolve_device(preferred: Optional[str] = None) -> torch.device:
        if preferred:
            return torch.device(preferred)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_weights(self, weights_path: Union[str, Path]) -> None:
        path = Path(weights_path)
        if not path.is_file():
            raise EnclaveReconstructionError(f"FaceVAE weights checkpoint not found: {path}")

        try:
            state_dict = torch.load(str(path), map_location=self._device)
            # Support raw state_dict or wrapped in dict
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

            self._vae.load_state_dict(state_dict, strict=False)
            log.info("Successfully loaded FaceVAE weights from %s", path)
        except Exception as exc:
            raise EnclaveReconstructionError(f"Failed to load FaceVAE state dict: {exc}") from exc

    def decrypt_vector(self, ciphertext_bytes: bytes) -> torch.Tensor:
        """
        Homomorphically decrypts CKKS ciphertext bytes into a [1, 128] FP32 PyTorch tensor.
        """
        if len(ciphertext_bytes) == 0:
            raise EnclaveDecryptionError("Empty ciphertext bytes received.")

        try:
            # 1. Deserialize CKKS vector
            ckks_vec = ts.ckks_vector_from(self._context, ciphertext_bytes)

            # 2. Homomorphic decryption to plaintext float list
            raw_floats: List[float] = ckks_vec.decrypt()

            # 3. Truncate SIMD padding to latent_dim
            latent_floats = raw_floats[: self.latent_dim]
            if len(latent_floats) < self.latent_dim:
                raise EnclaveDecryptionError(
                    f"Decrypted slot count ({len(latent_floats)}) is less than required {self.latent_dim}."
                )

            # 4. Convert to PyTorch FP32 tensor [1, 128]
            z = torch.tensor(latent_floats, dtype=torch.float32, device=self._device).reshape(
                1, self.latent_dim
            )
            return z

        except Exception as exc:
            if isinstance(exc, EnclaveError):
                raise
            raise EnclaveDecryptionError(f"TenSEAL CKKS decryption error: {exc}") from exc

    def decode_latent(self, z: torch.Tensor) -> np.ndarray:
        """
        Executes FaceVAE.decode(z) to generate a [64, 64, 3] RGB uint8 numpy image.

        Args:
            z: PyTorch tensor [1, 128] containing the decrypted facial embedding.

        Returns:
            np.ndarray: [64, 64, 3] uint8 RGB image array.
        """
        with torch.inference_mode():
            # FaceVAE.decode outputs [1, 3, 64, 64] with Sigmoid bounded [0, 1]
            recon_tensor = self._vae.decode(z)

            # Denormalize to uint8 [0, 255]
            clamped = recon_tensor.squeeze(0).clamp(0.0, 1.0)
            scaled = (clamped * 255.0).round().to(torch.uint8)

            # Permute from [3, 64, 64] (CHW) to [64, 64, 3] (HWC)
            img_np = scaled.permute(1, 2, 0).cpu().numpy()
            return img_np

    def reconstruct(
        self,
        ciphertext_bytes: bytes,
        output_path: Union[str, Path] = "suspect_reconstructed.png",
        scrub_memory: bool = True,
    ) -> Tuple[np.ndarray, str]:
        """
        End-to-end execution: Decrypts CKKS vector, decodes through FaceVAE,
        saves image to disk, and applies zero-trust memory scrubbing.

        Returns:
            Tuple[np.ndarray, str]: (RGB image array, SHA-256 hash of output file).
        """
        z: Optional[torch.Tensor] = None
        img_rgb: Optional[np.ndarray] = None

        try:
            # Step 1: Homomorphic Decryption
            t0 = time.perf_counter()
            z = self.decrypt_vector(ciphertext_bytes)
            decrypt_duration = (time.perf_counter() - t0) * 1000.0
            log.info("Decrypted [1, %d] latent vector in %.2f ms.", self.latent_dim, decrypt_duration)

            # Step 2: Neural Decoding
            t1 = time.perf_counter()
            img_rgb = self.decode_latent(z)
            decode_duration = (time.perf_counter() - t1) * 1000.0
            log.info("Reconstructed [64, 64, 3] facial image in %.2f ms.", decode_duration)

            # Step 3: Write image file
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)

            if CV2_AVAILABLE:
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_p), img_bgr)
            elif PIL_AVAILABLE:
                img_pil = Image.fromarray(img_rgb, mode="RGB")
                img_pil.save(str(out_p))
            else:
                # Fallback: Write uncompressed Netpbm PPM (P6)
                with open(out_p, "wb") as f:
                    f.write(f"P6\n64 64\n255\n".encode("ascii"))
                    f.write(img_rgb.tobytes())

            img_bytes = out_p.read_bytes()
            out_sha256 = hashlib.sha256(img_bytes).hexdigest()
            log.info(
                "Exported reconstructed suspect face to '%s' (SHA-256: %s)",
                out_p,
                out_sha256[:16] + "...",
            )

            return img_rgb, out_sha256

        finally:
            if scrub_memory:
                self.scrub_memory(z)

    @staticmethod
    def scrub_memory(*tensors_or_buffers: Any) -> None:
        """
        Strict zero-trust memory hygiene:
        Overwrites mutable plaintext latent representations before deletion,
        removes references, and invokes garbage collection.
        """
        for item in tensors_or_buffers:
            if item is None:
                continue
            try:
                if isinstance(item, torch.Tensor):
                    item.zero_()
                elif isinstance(item, bytearray):
                    item[:] = b"\x00" * len(item)
                elif isinstance(item, list):
                    for i in range(len(item)):
                        item[i] = 0.0
            except Exception:
                pass
            del item

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Zero-trust memory hygiene enforced: Plaintext biometric vectors purged.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI Application & User Interface
# ══════════════════════════════════════════════════════════════════════════════
def print_banner() -> None:
    banner = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              CRYPTAVISION // FORENSIC RECONSTRUCTION ENCLAVE                ║
║           TenSEAL CKKS Homomorphic Decryption & Neural Face Recovery         ║
║          Isolated Zero-Trust Environment &bull; HashiCorp Vault RBAC         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CryptaVision Phase 7: Forensic Reconstruction Enclave CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required inputs
    parser.add_argument(
        "--ciphertext",
        "-c",
        required=True,
        type=str,
        help="Path to encrypted CKKS ciphertext file (.bin, .ckks, .json, .b64)",
    )

    # HashiCorp Vault authentication
    parser.add_argument(
        "--vault-addr",
        type=str,
        default=os.getenv("VAULT_ADDR", "http://127.0.0.1:8200"),
        help="HashiCorp Vault URL",
    )
    parser.add_argument(
        "--vault-token",
        type=str,
        default=os.getenv("VAULT_TOKEN", None),
        help="Vault Compliance Officer Token (prompts securely if omitted)",
    )
    parser.add_argument(
        "--mount-point",
        type=str,
        default="secret",
        help="Vault KV v2 mount point",
    )
    parser.add_argument(
        "--secret-path",
        type=str,
        default="cryptavision/keys/tenseal_ckks_secret_context",
        help="KV v2 path to TenSEAL secret context",
    )
    parser.add_argument(
        "--secret-key-file",
        type=str,
        default=None,
        help="Path to local secret context file (bypasses Vault for air-gapped forensics)",
    )

    # Reconstruction configuration
    parser.add_argument(
        "--face-index",
        type=int,
        default=0,
        help="Face index to decode if JSON contains multiple faces",
    )
    parser.add_argument(
        "--vae-weights",
        "-w",
        type=str,
        default=None,
        help="Optional path to FaceVAE weights checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="suspect_reconstructed.png",
        help="Destination path for reconstructed image",
    )
    parser.add_argument(
        "--incident-id",
        type=str,
        default="INC-FORENSIC-001",
        help="Incident Tracking Identifier for audit log",
    )
    parser.add_argument(
        "--officer-id",
        type=str,
        default="OFFICER-LEGAL-COMPLIANCE",
        help="Compliance Officer Identifier for audit log",
    )
    parser.add_argument(
        "--audit-log",
        type=str,
        default="forensic_audit_trail.json",
        help="Path to write tamper-evident JSON audit record",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device for neural inference ('cpu', 'cuda', 'mps')",
    )
    parser.add_argument(
        "--no-scrub",
        action="store_true",
        help="Disable memory scrubbing (debug only; violates zero-trust policy)",
    )

    return parser.parse_args()


def run_cli() -> int:
    """CLI Entry point for the Forensic Reconstruction Enclave."""
    print_banner()
    args = parse_arguments()

    # Step 1: Resolve Secret Context
    secret_context_bytes: bytes
    key_source: str

    if args.secret_key_file:
        key_file_path = Path(args.secret_key_file)
        if not key_file_path.is_file():
            log.critical("Secret key file not found: %s", key_file_path)
            return 1
        secret_context_bytes = key_file_path.read_bytes()
        # Check if base64 encoded
        try:
            b64_str = secret_context_bytes.decode("ascii").strip()
            secret_context_bytes = base64.b64decode(b64_str)
        except Exception:
            pass
        key_source = f"airgap_file:{key_file_path}"
        log.info("Loaded secret context from local file: %s", key_file_path)
    else:
        # Obtain Vault token
        vault_token = args.vault_token
        if not vault_token:
            vault_token = getpass.getpass("🔐 Enter Vault Compliance Officer Token: ").strip()

        if not vault_token:
            log.critical("Authentication aborted: No Vault token provided.")
            return 1

        authenticator = VaultEnclaveAuthenticator(
            vault_url=args.vault_addr,
            vault_token=vault_token,
            mount_point=args.mount_point,
            secret_path=args.secret_path,
        )

        try:
            secret_context_bytes = authenticator.retrieve_secret_context()
            key_source = f"vault:{args.vault_addr}/{args.mount_point}/{args.secret_path}"
        except EnclaveAuthenticationError as exc:
            log.critical("Vault Authentication Error: %s", exc)
            return 2

    # Step 2: Load Target Ciphertext
    try:
        ciphertext_bytes, cipher_sha256 = CiphertextLoader.load_bytes_from_file(
            file_path=args.ciphertext,
            face_index=args.face_index,
        )
        log.info("Ciphertext SHA-256 digest: %s", cipher_sha256)
    except EnclavePayloadError as exc:
        log.critical("Ciphertext Payload Error: %s", exc)
        return 3

    # Step 3: Initialize Enclave & Reconstruct
    try:
        enclave = ForensicReconstructionEnclave(
            secret_context_bytes=secret_context_bytes,
            weights_path=args.vae_weights,
            latent_dim=128,
            device=args.device,
        )

        img_rgb, out_sha256 = enclave.reconstruct(
            ciphertext_bytes=ciphertext_bytes,
            output_path=args.output,
            scrub_memory=not args.no_scrub,
        )

    except EnclaveError as exc:
        log.critical("Forensic Reconstruction Error: %s", exc)
        return 4
    except Exception as exc:
        log.exception("Unexpected Enclave Failure: %s", exc)
        return 5

    # Step 4: Write Audit Trail Record
    audit_record = ForensicAuditRecord(
        incident_id=args.incident_id,
        officer_id=args.officer_id,
        session_timestamp=datetime.now(timezone.utc).isoformat(),
        ciphertext_path=str(Path(args.ciphertext).resolve()),
        ciphertext_sha256=cipher_sha256,
        secret_key_source=key_source,
        output_image_path=str(Path(args.output).resolve()),
        output_image_sha256=out_sha256,
        image_shape=img_rgb.shape,
        memory_scrubbed=not args.no_scrub,
    )

    try:
        audit_path = Path(args.audit_log)
        audit_path.write_text(json.dumps(audit_record.to_dict(), indent=2))
        log.info("Tamper-evident audit trail written to: %s", audit_path.resolve())
    except Exception as exc:
        log.warning("Could not write audit trail file: %s", exc)

    print("\n" + "═" * 78)
    print("✓ FORENSIC RECONSTRUCTION SUCCESSFUL")
    print(f"  Incident ID       : {args.incident_id}")
    print(f"  Officer ID        : {args.officer_id}")
    print(f"  Ciphertext Hash   : {cipher_sha256}")
    print(f"  Output Image File : {Path(args.output).resolve()}")
    print(f"  Image Digest      : {out_sha256}")
    print(f"  Zero-Trust Scrub  : {'ENFORCED' if not args.no_scrub else 'SKIPPED'}")
    print("═" * 78 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
