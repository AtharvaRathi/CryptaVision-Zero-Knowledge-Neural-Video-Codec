#!/usr/bin/env python3
"""
CryptaVision Phase 3 — Homomorphic Encryption Engine (CKKS)
============================================================
Provides mathematically precise homomorphic encryption for 128-dimensional
face latent vectors (`mu` from the Phase 2 FaceVAE) using the Microsoft SEAL
wrapper (TenSEAL).

Mathematical & Cryptographic Parameter Justification:
-----------------------------------------------------
1. Scheme: CKKS (Cheon-Kim-Kim-Song)
   - Specifically designed for floating-point complex/real arithmetic.
   - Ideal for ML embedding distance metrics, similarity calculations, and secure aggregation.

2. Polynomial Modulus Degree (N): 8192
   - Provides N/2 = 4096 SIMD vector slots.
   - 128-d latent fits comfortably within 4096 slots.
   - Together with the 200-bit coefficient modulus, strictly satisfies the 128-bit
     classical and quantum security level defined by the Homomorphic Encryption
     Standard (HomomorphicEncryption.org).

3. Coefficient Modulus Bit Sizes: [60, 40, 40, 60]
   - Total bit length: 200 bits <= 218 bits (maximum threshold for N=8192 at 128-bit security).
   - First 60-bit prime: Input scaling & initial noise ceiling.
   - Intermediate 40-bit primes: Allow 2 multiplicative levels without bootstrapping.
   - Final 60-bit prime: Preserves high precision upon decryption.

4. Scale: 2**40 (~1.0995e12)
   - Matches the 40-bit intermediate prime moduli for uniform rescale operations.
   - Ensures an encryption noise floor < 1e-6, maintaining full fidelity of
     FP32 latent representations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    ts = None  # type: ignore
    TENSEAL_AVAILABLE = False

log = logging.getLogger("CryptaVision.CryptoEngine")


class CryptoEngine:
    """
    Homomorphic Encryption Engine implementing CKKS for 128-d latent representations.

    Attributes:
        poly_modulus_degree (int): 8192
        coeff_mod_bit_sizes (List[int]): [60, 40, 40, 60]
        scale (float): 2**40
        latent_dim (int): 128
    """

    POLY_MODULUS_DEGREE: int = 8192
    COEFF_MOD_BIT_SIZES: List[int] = [60, 40, 40, 60]
    SCALE: float = float(2**40)
    DEFAULT_LATENT_DIM: int = 128

    def __init__(
        self,
        context: Optional[Union[bytes, Any]] = None,
        latent_dim: int = DEFAULT_LATENT_DIM,
    ):
        """
        Initializes the CKKS Homomorphic Encryption Engine.

        Args:
            context: Optional serialized context bytes or existing ts.Context.
                     If None, creates a fresh context with keypair generation.
            latent_dim: Dimensionality of latent embeddings (default: 128).
        """
        if not TENSEAL_AVAILABLE:
            raise ImportError(
                "TenSEAL is required for CryptoEngine. "
                "Install it using: pip install tenseal>=0.3.14"
            )

        self.latent_dim = latent_dim

        if context is None:
            log.info(
                "Initializing fresh TenSEAL CKKS Context (N=%d, bits=%s, scale=2^40)",
                self.POLY_MODULUS_DEGREE,
                self.COEFF_MOD_BIT_SIZES,
            )
            # Create fresh CKKS context
            self._context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.POLY_MODULUS_DEGREE,
                coeff_mod_bit_sizes=self.COEFF_MOD_BIT_SIZES,
            )
            self._context.global_scale = self.SCALE

            # Generate evaluation keys for homomorphic operations
            self._context.generate_galois_keys()
            self._context.generate_relin_keys()
        elif isinstance(context, bytes):
            log.info("Loading TenSEAL context from %d serialized bytes", len(context))
            self._context = ts.context_from(context)
        elif hasattr(context, "is_private"):
            # Existing TenSEAL context object
            self._context = context
        else:
            raise TypeError(
                f"Unsupported context type: {type(context)}. "
                "Expected bytes, tenseal.Context, or None."
            )

    @property
    def context(self) -> Any:
        """Returns the underlying TenSEAL Context object."""
        return self._context

    @property
    def is_private(self) -> bool:
        """Returns True if the context contains the secret decryption key."""
        return bool(self._context.is_private())

    def serialize_secret_context(self) -> bytes:
        """
        Serializes the complete context including the private key.
        CRITICAL: This payload must only be stored in secure vaults (HashiCorp Vault).

        Returns:
            bytes: Serialized private context.
        """
        if not self.is_private:
            raise ValueError("Cannot serialize private context: secret key not present.")
        return bytes(self._context.serialize(save_secret_key=True))

    def serialize_public_context(self) -> bytes:
        """
        Serializes the public context (evaluation keys only, no secret key).
        Safe for distribution to untrusted analytics worker nodes.

        Returns:
            bytes: Serialized public evaluation context.
        """
        return bytes(self._context.serialize(save_secret_key=False))

    @classmethod
    def from_serialized_context(
        cls,
        context_bytes: bytes,
        latent_dim: int = DEFAULT_LATENT_DIM,
    ) -> CryptoEngine:
        """Factory method to instantiate CryptoEngine from serialized bytes."""
        return cls(context=context_bytes, latent_dim=latent_dim)

    def encrypt_latent(self, tensor: torch.Tensor) -> bytes:
        """
        Encrypts a [1, 128] PyTorch latent vector into a serialized CKKS ciphertext.

        Strict Processing Flow:
          1. Detaches tensor from PyTorch autograd graph.
          2. Moves tensor to CPU and casts to float32.
          3. Validates that element count matches `self.latent_dim` (128).
          4. Flattens tensor into a 1D Python list.
          5. Encrypts into a TenSEAL CKKSVector under current context.
          6. Serializes the CKKSVector into raw bytes.

        Args:
            tensor: PyTorch tensor with shape [1, 128] or [128].

        Returns:
            bytes: Serialized encrypted CKKS ciphertext.
        """
        if not isinstance(tensor, torch.Tensor):
            if isinstance(tensor, (np.ndarray, list)):
                tensor = torch.as_tensor(tensor, dtype=torch.float32)
            else:
                raise TypeError(
                    f"Expected tensor of type torch.Tensor, got {type(tensor).__name__}"
                )

        if tensor.numel() != self.latent_dim:
            raise ValueError(
                f"Dimension mismatch: Expected {self.latent_dim} elements, "
                f"got shape {tuple(tensor.shape)} ({tensor.numel()} elements)"
            )

        # 1-3. Detach, CPU, float32, flatten to 1D list
        flat_list: List[float] = (
            tensor.detach()
            .cpu()
            .flatten()
            .to(torch.float32)
            .tolist()
        )

        # 4. Homomorphic encryption into CKKSVector
        ckks_vec = ts.ckks_vector(self._context, flat_list)

        # 5. Serialize ciphertext
        return bytes(ckks_vec.serialize())

    def decrypt_latent(self, encrypted_bytes: bytes) -> torch.Tensor:
        """
        Decrypts serialized CKKS ciphertext back to a [1, 128] PyTorch tensor.

        Strict Security & Numerical Guarantees:
          1. Verifies that the context holds the private secret key.
          2. Deserializes the bytes into a CKKSVector.
          3. Decrypts the plaintext slot values.
          4. Slices to exact `latent_dim` to remove SIMD slot padding.
          5. Reconstructs a PyTorch FP32 tensor with shape [1, 128].

        Args:
            encrypted_bytes: Serialized CKKS ciphertext bytes.

        Returns:
            torch.Tensor: Decrypted tensor with shape [1, 128], dtype float32.
        """
        if not self.is_private:
            raise PermissionError(
                "Decryption failed: Current TenSEAL context does not contain the secret key. "
                "Decryption requires the private key retrieved from secure storage."
            )

        if not isinstance(encrypted_bytes, (bytes, bytearray)) or len(encrypted_bytes) == 0:
            raise ValueError("encrypted_bytes must be a non-empty bytes object")

        # Deserialize ciphertext vector
        ckks_vec = ts.ckks_vector_from(self._context, bytes(encrypted_bytes))

        # Decrypt to plaintext list
        decrypted_slots: List[float] = ckks_vec.decrypt()

        # Slice to latent_dim and reshape to [1, 128]
        tensor = torch.tensor(
            decrypted_slots[:self.latent_dim],
            dtype=torch.float32,
        ).reshape(1, self.latent_dim)

        return tensor

    def compute_reconstruction_metrics(
        self,
        original: torch.Tensor,
        decrypted: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Calculates precision fidelity metrics between original and decrypted vectors.
        Because CKKS is an approximate HE scheme, this measures the encryption noise floor.

        Args:
            original: Original PyTorch tensor before encryption.
            decrypted: Decrypted PyTorch tensor.

        Returns:
            Dict containing 'max_abs_error', 'mean_abs_error', 'mse', 'snr_db'.
        """
        orig_flat = original.detach().cpu().flatten().float()
        dec_flat = decrypted.detach().cpu().flatten().float()

        diff = orig_flat - dec_flat
        abs_err = torch.abs(diff)

        max_err = float(torch.max(abs_err))
        mae = float(torch.mean(abs_err))
        mse = float(torch.mean(diff ** 2))

        # Signal-to-noise ratio in decibels
        signal_pwr = float(torch.mean(orig_flat ** 2))
        noise_pwr = max(mse, 1e-15)
        snr_db = float(10 * np.log10(signal_pwr / noise_pwr))

        return {
            "max_abs_error": max_err,
            "mean_abs_error": mae,
            "mse": mse,
            "snr_db": snr_db,
        }
