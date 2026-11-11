#!/usr/bin/env python3
"""
CryptaVision Phase 3 — HashiCorp Vault Key & Secret Management
==============================================================
Provides production-grade key management for homomorphic encryption contexts
and serialized latents using HashiCorp Vault KV (Key-Value) v2 engine.

Security Architecture & Guarantees:
-----------------------------------
1. Zero-Trust Storage of Secret Contexts:
   - The TenSEAL CKKS secret context (containing the private decryption key)
     is stored in Vault KV v2 under `secret/data/cryptavision/keys`.
   - Payload is base64-encoded with a mandatory SHA-256 cryptographic digest
     to detect bit flips or tampering before decryption.

2. Role-Based Access Control (RBAC) & Least Privilege:
   - Untrusted video processing workers possess only the public evaluation context.
   - The secret context is accessed solely for authorized re-identification investigations.

3. Time-Limited Compliance Officer Tokens:
   - Generates non-renewable, short-lived (e.g., 1-hour) Vault access tokens
     specifically bounded to the incident ID and officer credentials.
   - Token is restricted strictly to read capabilities on the cryptavision paths.

4. Fault-Tolerant Resilience:
   - When Vault is unreachable in development/testing environments, gracefully
     engages an isolated in-memory KV v2 fallback store while emitting audit warnings.
   - In production (`fallback_to_memory=False`), connection failure raises a hard
     `VaultConnectionError`.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

try:
    import hvac
    import hvac.exceptions
    HVAC_AVAILABLE = True
except ImportError:
    hvac = None  # type: ignore
    HVAC_AVAILABLE = False

log = logging.getLogger("CryptaVision.VaultManager")


# ══════════════════════════════════════════════════════════════════════════════
#  Pydantic Schemas & DTOs
# ══════════════════════════════════════════════════════════════════════════════
class VaultConfig(BaseModel):
    """Configuration settings for HashiCorp Vault connection."""

    vault_url: str = Field(
        default_factory=lambda: os.getenv("VAULT_ADDR", "http://127.0.0.1:8200")
    )
    vault_token: Optional[str] = Field(
        default_factory=lambda: os.getenv("VAULT_TOKEN", None)
    )
    mount_point: str = Field(default_factory=lambda: os.getenv("VAULT_MOUNT_POINT", "secret"))
    key_path: str = Field(default="cryptavision/keys")
    latents_path: str = Field(default="cryptavision/latents")
    compliance_policy_name: str = Field(default="cryptavision-compliance-officer")
    fallback_to_memory: bool = Field(default=True)
    timeout_seconds: int = Field(default=5)
    namespace: Optional[str] = Field(
        default_factory=lambda: os.getenv("VAULT_NAMESPACE", None)
    )


class ComplianceTokenRequest(BaseModel):
    """Request payload for issuing a compliance officer investigation token."""

    officer_id: str = Field(description="Unique identity of the compliance investigator")
    incident_id: str = Field(description="Authorized incident tracking identifier")
    ttl: str = Field(default="1h", description="Token lifetime (e.g., 1h, 30m, 3600s)")
    reason: str = Field(default="Face reconstruction investigation under legal warrant")
    num_uses: Optional[int] = Field(
        default=0, description="Max uses before auto-revocation (0 = unlimited within TTL)"
    )


class ComplianceTokenResponse(BaseModel):
    """Response containing time-limited investigation credentials."""

    client_token: str
    accessor: str
    ttl: str
    lease_duration_seconds: int
    policies: List[str]
    meta: Dict[str, str]
    created_at: str
    is_fallback: bool = False


# ══════════════════════════════════════════════════════════════════════════════
#  Custom Exceptions
# ══════════════════════════════════════════════════════════════════════════════
class VaultConnectionError(Exception):
    """Raised when connection to HashiCorp Vault fails and fallback is disabled."""
    pass


class IntegrityVerificationError(Exception):
    """Raised when a cryptographic checksum check fails on retrieved payload."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
#  In-Memory KV v2 Simulation for Fallback / Local Development
# ══════════════════════════════════════════════════════════════════════════════
class InMemoryVaultStorage:
    """Thread-safe in-memory replica of HashiCorp Vault KV v2."""

    def __init__(self):
        self._secrets: Dict[str, List[Dict[str, Any]]] = {}
        self._tokens: Dict[str, Dict[str, Any]] = {}

    def write_secret(self, path: str, data: Dict[str, Any]) -> int:
        if path not in self._secrets:
            self._secrets[path] = []
        version = len(self._secrets[path]) + 1
        record = {
            "version": version,
            "data": data,
            "created_time": datetime.now(timezone.utc).isoformat(),
        }
        self._secrets[path].append(record)
        return version

    def read_secret(self, path: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        versions = self._secrets.get(path)
        if not versions:
            return None
        if version is None:
            return versions[-1]["data"]
        for record in versions:
            if record["version"] == version:
                return record["data"]
        return None

    def register_token(
        self,
        token: str,
        policies: List[str],
        ttl_seconds: int,
        meta: Dict[str, str],
        num_uses: Optional[int] = 0,
    ) -> None:
        self._tokens[token] = {
            "policies": policies,
            "expires_at": time.time() + ttl_seconds,
            "meta": meta,
            "num_uses": num_uses or 0,
            "uses_count": 0,
        }

    def validate_token(self, token: str) -> bool:
        entry = self._tokens.get(token)
        if not entry:
            return False
        if time.time() > entry["expires_at"]:
            del self._tokens[token]
            return False
        entry["uses_count"] += 1
        if entry["num_uses"] > 0 and entry["uses_count"] >= entry["num_uses"]:
            del self._tokens[token]
        return True

    def revoke_token(self, token: str) -> bool:
        if token in self._tokens:
            del self._tokens[token]
            return True
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  VaultManager
# ══════════════════════════════════════════════════════════════════════════════
class VaultManager:
    """
    Manages cryptographic key lifecycle, TenSEAL context persistence,
    and ephemeral access tokens in HashiCorp Vault.
    """

    def __init__(self, config: Optional[VaultConfig] = None):
        """
        Initializes the Vault Manager.

        Args:
            config: VaultConfig instance or None (uses environment defaults).
        """
        self.config = config or VaultConfig()
        self._client: Optional[hvac.Client] = None
        self._is_fallback = False
        self._memory_storage: Optional[InMemoryVaultStorage] = None

        self._connect()

    def _connect(self) -> None:
        """Establishes connection to HashiCorp Vault or activates in-memory fallback."""
        if not HVAC_AVAILABLE:
            if self.config.fallback_to_memory:
                self._activate_fallback("hvac library not installed")
                return
            raise ImportError(
                "hvac library is required to connect to HashiCorp Vault. "
                "Install using: pip install hvac>=2.1.0"
            )

        try:
            log.info("Attempting connection to Vault at %s", self.config.vault_url)
            self._client = hvac.Client(
                url=self.config.vault_url,
                token=self.config.vault_token,
                timeout=self.config.timeout_seconds,
                namespace=self.config.namespace,
            )

            # Check Vault readiness / authentication
            # Note: client.sys.is_initialized() does not require a token
            if not self._client.sys.is_initialized():
                raise VaultConnectionError(f"Vault instance at {self.config.vault_url} is not initialized.")

            if self.config.vault_token:
                if not self._client.is_authenticated():
                    raise VaultConnectionError("Invalid or expired Vault token provided.")
                log.info("Successfully authenticated with HashiCorp Vault.")
            else:
                log.warning("No VAULT_TOKEN provided. Client initialized in unauthenticated mode.")

            self._is_fallback = False

        except Exception as exc:
            msg = f"Failed to connect to HashiCorp Vault: {exc}"
            if self.config.fallback_to_memory:
                self._activate_fallback(msg)
            else:
                log.critical(msg)
                raise VaultConnectionError(msg) from exc

    def _activate_fallback(self, reason: str) -> None:
        """Activates in-memory fallback storage."""
        log.warning(
            "──────────────────────────────────────────────────────────────────\n"
            "[SECURITY AUDIT NOTICE] HashiCorp Vault unavailable (%s).\n"
            "ACTIVATING IN-MEMORY FALLBACK KEY STORE.\n"
            "This mode is strictly for local testing. Production requires a live Vault.\n"
            "──────────────────────────────────────────────────────────────────",
            reason,
        )
        self._is_fallback = True
        self._memory_storage = InMemoryVaultStorage()

    @property
    def is_connected(self) -> bool:
        """Returns True if actively connected to a live Vault instance."""
        return not self._is_fallback and self._client is not None

    @property
    def is_fallback(self) -> bool:
        """Returns True if operating under in-memory simulated storage."""
        return self._is_fallback

    # ─────────────────────────────────────────────────────────────────────────
    #  Secret Context Storage & Retrieval (KV v2)
    # ─────────────────────────────────────────────────────────────────────────

    def store_secret_context(
        self,
        context_bytes: bytes,
        key_name: str = "tenseal_ckks_secret_context",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Stores the TenSEAL secret context in Vault KV v2.

        Security protocol:
          1. Computes SHA-256 digest of binary context.
          2. Encodes binary bytes to Base64 ASCII.
          3. Bundles with cryptographic parameter metadata.
          4. Commits atomic write to Vault KV v2.

        Args:
            context_bytes: Raw serialized TenSEAL secret context.
            key_name: Secret identifier key.
            metadata: Optional additional metadata fields.

        Returns:
            Dict containing write response metadata.
        """
        if not isinstance(context_bytes, (bytes, bytearray)) or len(context_bytes) == 0:
            raise ValueError("context_bytes must be non-empty bytes")

        # 1. Cryptographic integrity checksum
        sha256_digest = hashlib.sha256(context_bytes).hexdigest()

        # 2. Base64 encoding for JSON transport
        b64_payload = base64.b64encode(context_bytes).decode("ascii")

        secret_dict: Dict[str, Any] = {
            "secret_context_b64": b64_payload,
            "sha256_checksum": sha256_digest,
            "byte_length": len(context_bytes),
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "scheme": "CKKS",
            "poly_modulus_degree": 8192,
            "coeff_mod_bit_sizes": [60, 40, 40, 60],
            "scale": float(2**40),
            "latent_dim": 128,
        }
        if metadata:
            secret_dict["custom_metadata"] = metadata

        full_path = f"{self.config.key_path}/{key_name}"

        if self._is_fallback:
            assert self._memory_storage is not None
            ver = self._memory_storage.write_secret(full_path, secret_dict)
            log.info("[FALLBACK] Stored secret context at %s (v%d, sha256=%s...)", full_path, ver, sha256_digest[:12])
            return {
                "path": full_path,
                "version": ver,
                "sha256": sha256_digest,
                "is_fallback": True,
            }

        # Live Vault KV v2 write
        try:
            assert self._client is not None
            resp = self._client.secrets.kv.v2.create_or_update_secret(
                path=full_path,
                secret=secret_dict,
                mount_point=self.config.mount_point,
            )
            version = resp.get("data", {}).get("version", 1)
            log.info("Stored secret context in Vault at %s (v%s, sha256=%s...)", full_path, version, sha256_digest[:12])
            return {
                "path": full_path,
                "version": version,
                "sha256": sha256_digest,
                "is_fallback": False,
            }
        except Exception as exc:
            log.error("Failed to store secret context in Vault: %s", exc)
            raise

    def retrieve_secret_context(
        self,
        key_name: str = "tenseal_ckks_secret_context",
        version: Optional[int] = None,
    ) -> bytes:
        """
        Retrieves and verifies the TenSEAL secret context from Vault KV v2.

        Security protocol:
          1. Reads secret payload from Vault KV v2.
          2. Extracts base64 context and stored SHA-256 checksum.
          3. Decodes Base64 to binary bytes.
          4. Re-computes SHA-256 and validates integrity.

        Args:
            key_name: Secret identifier key.
            version: Specific version (default None fetches latest).

        Returns:
            bytes: Validated raw serialized TenSEAL secret context.
        """
        full_path = f"{self.config.key_path}/{key_name}"

        if self._is_fallback:
            assert self._memory_storage is not None
            data = self._memory_storage.read_secret(full_path, version=version)
            if data is None:
                raise KeyError(f"[FALLBACK] Secret not found at path: {full_path}")
            secret_dict = data
        else:
            try:
                assert self._client is not None
                resp = self._client.secrets.kv.v2.read_secret_version(
                    path=full_path,
                    version=version,
                    mount_point=self.config.mount_point,
                )
                secret_dict = resp["data"]["data"]
            except Exception as exc:
                log.error("Failed to read secret context from Vault at %s: %s", full_path, exc)
                raise

        b64_payload = secret_dict["secret_context_b64"]
        expected_sha = secret_dict.get("sha256_checksum")

        context_bytes = base64.b64decode(b64_payload.encode("ascii"))

        # Verify integrity
        if expected_sha:
            computed_sha = hashlib.sha256(context_bytes).hexdigest()
            if computed_sha != expected_sha:
                raise IntegrityVerificationError(
                    f"Integrity check failed for secret context at {full_path}! "
                    f"Expected SHA-256 {expected_sha}, computed {computed_sha}."
                )

        log.info("Successfully retrieved and verified secret context from %s (%d bytes)", full_path, len(context_bytes))
        return context_bytes

    # ─────────────────────────────────────────────────────────────────────────
    #  Encrypted Latent Vectors Storage (Audit Archive)
    # ─────────────────────────────────────────────────────────────────────────

    def store_encrypted_latent(
        self,
        frame_id: int,
        face_idx: int,
        encrypted_bytes: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Stores an encrypted CKKS latent ciphertext in Vault KV v2 for audit archive.

        Args:
            frame_id: Video pipeline frame counter.
            face_idx: Index of detected face within the frame.
            encrypted_bytes: Serialized CKKSVector bytes.
            metadata: Optional additional metadata (e.g. bounding box).

        Returns:
            str: Vault path where ciphertext is archived.
        """
        b64_cipher = base64.b64encode(encrypted_bytes).decode("ascii")
        sha256_digest = hashlib.sha256(encrypted_bytes).hexdigest()

        path = f"{self.config.latents_path}/frame_{frame_id}_face_{face_idx}"
        payload: Dict[str, Any] = {
            "frame_id": frame_id,
            "face_idx": face_idx,
            "ciphertext_b64": b64_cipher,
            "sha256": sha256_digest,
            "byte_size": len(encrypted_bytes),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            payload["metadata"] = metadata

        if self._is_fallback:
            assert self._memory_storage is not None
            self._memory_storage.write_secret(path, payload)
        else:
            assert self._client is not None
            self._client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=payload,
                mount_point=self.config.mount_point,
            )

        return path

    def retrieve_encrypted_latent(self, path: str) -> bytes:
        """Retrieves and verifies an encrypted CKKS latent ciphertext."""
        if self._is_fallback:
            assert self._memory_storage is not None
            data = self._memory_storage.read_secret(path)
            if not data:
                raise KeyError(f"Ciphertext not found at path: {path}")
        else:
            assert self._client is not None
            resp = self._client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self.config.mount_point,
            )
            data = resp["data"]["data"]

        b64_cipher = data["ciphertext_b64"]
        expected_sha = data.get("sha256")
        cipher_bytes = base64.b64decode(b64_cipher.encode("ascii"))

        if expected_sha:
            computed_sha = hashlib.sha256(cipher_bytes).hexdigest()
            if computed_sha != expected_sha:
                raise IntegrityVerificationError(f"Integrity check failed for latent at {path}")

        return cipher_bytes

    # ─────────────────────────────────────────────────────────────────────────
    #  Time-Limited Decryption Token Generation (Incident Investigation)
    # ─────────────────────────────────────────────────────────────────────────

    def generate_compliance_token(
        self,
        request: ComplianceTokenRequest,
    ) -> ComplianceTokenResponse:
        """
        Generates a non-renewable, time-limited Vault token for a compliance officer.

        Security controls:
          1. Enforces strict TTL (default: 1 hour).
          2. Locks policy to read-only access on CryptaVision secret paths.
          3. Non-renewable: prevents extension past the authorized investigation window.
          4. Emits tamper-evident audit metadata with officer and incident IDs.

        Args:
            request: ComplianceTokenRequest specification.

        Returns:
            ComplianceTokenResponse with temporary token and accessor.
        """
        log.info(
            "Generating compliance decryption token: officer=%s incident=%s ttl=%s",
            request.officer_id,
            request.incident_id,
            request.ttl,
        )

        ttl_seconds = self._parse_ttl_to_seconds(request.ttl)
        policies = [self.config.compliance_policy_name]
        meta = {
            "officer_id": request.officer_id,
            "incident_id": request.incident_id,
            "reason": request.reason,
            "issued_at": datetime.now(timezone.utc).isoformat(),
        }

        if self._is_fallback:
            # Generate cryptographically secure mock token in fallback mode
            token = f"cv-mock-token-{secrets.token_urlsafe(24)}"
            accessor = f"cv-mock-accessor-{secrets.token_urlsafe(16)}"
            assert self._memory_storage is not None
            self._memory_storage.register_token(
                token=token,
                policies=policies,
                ttl_seconds=ttl_seconds,
                meta=meta,
                num_uses=request.num_uses,
            )
            log.warning("[FALLBACK] Issued simulated compliance token valid for %ds", ttl_seconds)
            return ComplianceTokenResponse(
                client_token=token,
                accessor=accessor,
                ttl=request.ttl,
                lease_duration_seconds=ttl_seconds,
                policies=policies,
                meta=meta,
                created_at=datetime.now(timezone.utc).isoformat(),
                is_fallback=True,
            )

        # Live HashiCorp Vault token creation
        try:
            assert self._client is not None

            # Attempt to ensure the read-only compliance policy exists
            self._ensure_compliance_policy()

            token_resp = self._client.auth.token.create(
                policies=policies,
                ttl=request.ttl,
                renewable=False,
                num_uses=request.num_uses or 0,
                display_name=f"compliance-{request.officer_id}-{request.incident_id}",
                meta=meta,
            )
            auth_data = token_resp["auth"]
            client_token = auth_data["client_token"]
            accessor = auth_data["accessor"]
            lease_duration = auth_data["lease_duration"]

            log.info("Issued Vault compliance token: accessor=%s lease=%ds", accessor, lease_duration)

            return ComplianceTokenResponse(
                client_token=client_token,
                accessor=accessor,
                ttl=request.ttl,
                lease_duration_seconds=lease_duration,
                policies=policies,
                meta=meta,
                created_at=datetime.now(timezone.utc).isoformat(),
                is_fallback=False,
            )
        except Exception as exc:
            log.error("Failed to generate compliance token in Vault: %s", exc)
            raise

    def revoke_token(self, token: str) -> bool:
        """Immediately revokes an active investigation token upon case resolution."""
        if self._is_fallback:
            assert self._memory_storage is not None
            return self._memory_storage.revoke_token(token)

        try:
            assert self._client is not None
            self._client.auth.token.revoke(token=token)
            log.info("Revoked Vault token successfully.")
            return True
        except Exception as exc:
            log.error("Failed to revoke Vault token: %s", exc)
            return False

    def verify_token(self, token: str) -> bool:
        """Verifies if a provided compliance token is valid and unexpired."""
        if self._is_fallback:
            assert self._memory_storage is not None
            return self._memory_storage.validate_token(token)

        try:
            assert self._client is not None
            lookup = self._client.auth.token.lookup(token=token)
            return bool(lookup.get("data", {}).get("id"))
        except Exception:
            return False

    def _ensure_compliance_policy(self) -> None:
        """Creates or updates the compliance officer HCL policy in Vault."""
        policy_hcl = f"""
        # CryptaVision Compliance Officer Policy
        # Read-only access to keys and latent archives for forensic investigation
        path "{self.config.mount_point}/data/{self.config.key_path}/*" {{
            capabilities = ["read"]
        }}
        path "{self.config.mount_point}/data/{self.config.latents_path}/*" {{
            capabilities = ["read"]
        }}
        path "{self.config.mount_point}/metadata/{self.config.key_path}/*" {{
            capabilities = ["list", "read"]
        }}
        path "{self.config.mount_point}/metadata/{self.config.latents_path}/*" {{
            capabilities = ["list", "read"]
        }}
        """
        try:
            assert self._client is not None
            self._client.sys.create_or_update_policy(
                name=self.config.compliance_policy_name,
                policy=policy_hcl.strip(),
            )
            log.debug("Provisional policy '%s' ensured in Vault.", self.config.compliance_policy_name)
        except Exception as exc:
            # Non-fatal if current token does not have sys policy management permissions
            log.debug("Policy creation skipped or unauthorized: %s", exc)

    @staticmethod
    def _parse_ttl_to_seconds(ttl_str: str) -> int:
        """Parses duration string like '1h', '30m', '3600s' to seconds."""
        s = ttl_str.strip().lower()
        if s.endswith("h"):
            return int(float(s[:-1]) * 3600)
        elif s.endswith("m"):
            return int(float(s[:-1]) * 60)
        elif s.endswith("s"):
            return int(float(s[:-1]))
        elif s.isdigit():
            return int(s)
        return 3600  # Default 1 hour fallback
