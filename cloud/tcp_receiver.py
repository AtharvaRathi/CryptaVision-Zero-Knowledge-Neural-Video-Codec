#!/usr/bin/env python3
"""
CryptaVision Phase 6 — Cloud Asynchronous TCP Receiver
======================================================
Asynchronous TCP ingestion server designed to run alongside or within the
Phase 4 FastAPI application.

Packet Processing Architecture:
-------------------------------
1. Binary Packet Protocol Unpacking:
   - Header (9 bytes):
     [4-byte Magic Number 0x43525950]
     [4-byte Payload Length (network byte order)]
     [1-byte Payload Type: 0x01 (JSON), 0x02 (CKKS)]
   - Payload (Length bytes):
     - Type 0x01: UTF-8 JSON telemetry (YOLO pose keypoints & CLIP embeddings).
     - Type 0x02: TenSEAL CKKS ciphertext & face bounding box envelope.

2. In-Memory Routing:
   - Unpacked JSON telemetry is dispatched directly to `StreamABroadcaster`
     for real-time delivery to Server-Sent Events (SSE) and WebSockets.
   - Unpacked CKKS ciphertexts are ingested into the Vault-protected
     `CloudLatentStore` for forensic extraction via `GET /analytics/stream-b`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [CloudTCP]: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("CloudTCP")

# Binary Protocol Constants
CRYPTA_MAGIC_NUMBER = 0x43525950  # ASCII 'CRYP'
HEADER_FORMAT = "!IIB"            # uint32 magic, uint32 length, uint8 type
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 9 bytes

TYPE_JSON = 0x01
TYPE_CKKS = 0x02


# ══════════════════════════════════════════════════════════════════════════════
#  Cloud Latent Ingestion Record & Datastore
# ══════════════════════════════════════════════════════════════════════════════
class CloudLatentRecord:
    """Thread-safe representation of an encrypted CKKS face latent received from the edge."""

    def __init__(
        self,
        frame_id: int,
        face_idx: int,
        box: List[int],
        pts_ns: int,
        ckks_bytes: bytes,
    ):
        self.frame_id = frame_id
        self.face_idx = face_idx
        self.box = box
        self.pts_ns = pts_ns
        self.ckks_bytes = ckks_bytes
        self.sha256 = hashlib.sha256(ckks_bytes).hexdigest()
        self.received_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "face_idx": self.face_idx,
            "box": self.box,
            "pts_ns": self.pts_ns,
            "ckks_bytes_b64": base64.b64encode(self.ckks_bytes).decode("ascii"),
            "byte_size": len(self.ckks_bytes),
            "sha256": self.sha256,
            "received_at": self.received_at,
        }


class CloudLatentStore:
    """Thread-safe ring buffer for received CKKS face latents."""

    def __init__(self, max_size: int = 256):
        self._records: List[CloudLatentRecord] = []
        self._lock = asyncio.Lock()
        self._max_size = max_size

    async def push(self, record: CloudLatentRecord) -> None:
        async with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_size:
                self._records = self._records[-self._max_size:]

    async def drain(self) -> List[CloudLatentRecord]:
        """Atomically drains collected encrypted latents."""
        async with self._lock:
            out, self._records = self._records, []
        return out

    async def count(self) -> int:
        async with self._lock:
            return len(self._records)


# Global Singleton for Cloud Ingestion
cloud_latent_store = CloudLatentStore()


# ══════════════════════════════════════════════════════════════════════════════
#  Asynchronous TCP Server Implementation
# ══════════════════════════════════════════════════════════════════════════════
class CloudTCPReceiver:
    """
    Asynchronous TCP Server unpacking binary telemetry from the edge C++ sink.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5555,
        on_json_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        latent_store: Optional[CloudLatentStore] = None,
    ):
        self.host = host
        self.port = port
        self.on_json_callback = on_json_callback
        self.latent_store = latent_store or cloud_latent_store
        self.server: Optional[asyncio.Server] = None
        self.is_running = False

        # Metrics
        self.total_packets = 0
        self.json_packets = 0
        self.ckks_packets = 0
        self.bytes_received = 0

    async def start(self) -> None:
        """Starts the asynchronous TCP listener."""
        self.server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
        )
        self.is_running = True
        addrs = ", ".join(str(sock.getsockname()) for sock in self.server.sockets)
        log.info("Cloud TCP Ingestion Server listening on %s", addrs)

    async def stop(self) -> None:
        """Stops the TCP server and closes client connections."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.is_running = False
            log.info("Cloud TCP Ingestion Server stopped.")

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Processes binary packets from an edge Jetson node connection."""
        peer = writer.get_extra_info("peername")
        log.info("Edge Jetson client connected from %s", peer)

        try:
            while True:
                # 1. Read 9-byte binary header
                try:
                    header_bytes = await reader.readexactly(HEADER_SIZE)
                except asyncio.IncompleteReadError:
                    log.info("Edge client %s disconnected (EOF).", peer)
                    break

                magic, length, payload_type = struct.unpack(HEADER_FORMAT, header_bytes)

                # 2. Validate Magic Number
                if magic != CRYPTA_MAGIC_NUMBER:
                    log.warning(
                        "Protocol Error: Invalid magic number 0x%08X (expected 0x%08X). Closing client %s.",
                        magic, CRYPTA_MAGIC_NUMBER, peer,
                    )
                    break

                # 3. Read exact payload bytes
                try:
                    payload = await reader.readexactly(length)
                except asyncio.IncompleteReadError:
                    log.warning("Incomplete payload from %s (expected %d bytes).", peer, length)
                    break

                self.total_packets += 1
                self.bytes_received += (HEADER_SIZE + length)

                # 4. Route payload based on type
                if payload_type == TYPE_JSON:
                    await self._process_json_packet(payload)
                elif payload_type == TYPE_CKKS:
                    await self._process_ckks_packet(payload)
                else:
                    log.warning("Unknown payload type: 0x%02X from %s", payload_type, peer)

        except Exception as exc:
            log.error("Exception handling edge client %s: %s", peer, exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _process_json_packet(self, payload: bytes) -> None:
        """Processes Stream A Semantic JSON metadata."""
        self.json_packets += 1
        try:
            text = payload.decode("utf-8")
            data = json.loads(text)

            # Dispatch to callback (e.g., StreamABroadcaster for SSE)
            if self.on_json_callback:
                self.on_json_callback(data)
            else:
                log.debug("Stream A Frame #%s: %d humans detected", data.get("frame_id"), data.get("num_humans", 0))

        except Exception as exc:
            log.error("Failed to parse Stream A JSON packet: %s", exc)

    async def _process_ckks_packet(self, payload: bytes) -> None:
        """Processes Stream B TenSEAL CKKS ciphertext."""
        self.ckks_packets += 1
        try:
            # Check for envelope prefix: [4-byte env_len][envelope JSON][CKKS bytes]
            if len(payload) > 4:
                env_len = int.from_bytes(payload[:4], byteorder="big")
                if 4 + env_len <= len(payload):
                    env_json_bytes = payload[4:4 + env_len]
                    env_data = json.loads(env_json_bytes.decode("utf-8"))
                    ckks_bytes = payload[4 + env_len:]

                    record = CloudLatentRecord(
                        frame_id=env_data.get("frame_id", 0),
                        face_idx=env_data.get("face_idx", 0),
                        box=env_data.get("box", [0, 0, 0, 0]),
                        pts_ns=env_data.get("pts_ns", 0),
                        ckks_bytes=ckks_bytes,
                    )
                    await self.latent_store.push(record)
                    log.info(
                        "Stored CKKS Latent: Frame #%d Face #%d (%d bytes, sha256=%s...)",
                        record.frame_id, record.face_idx, len(ckks_bytes), record.sha256[:12],
                    )
                    return

            # Fallback for raw CKKS bytes without envelope
            record = CloudLatentRecord(
                frame_id=self.ckks_packets,
                face_idx=0,
                box=[0, 0, 0, 0],
                pts_ns=0,
                ckks_bytes=payload,
            )
            await self.latent_store.push(record)

        except Exception as exc:
            log.error("Failed to parse Stream B CKKS packet: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone Runner
# ══════════════════════════════════════════════════════════════════════════════
async def async_main():
    parser = argparse.ArgumentParser(description="CryptaVision Cloud TCP Receiver")
    parser.add_argument("--host", default="0.0.0.0", help="TCP listen address")
    parser.add_argument("--port", type=int, default=5555, help="TCP listen port")
    args = parser.parse_args()

    receiver = CloudTCPReceiver(host=args.host, port=args.port)
    await receiver.start()

    try:
        while True:
            await asyncio.sleep(10.0)
            log.info(
                "Ingestion Stats: packets=%d (json=%d, ckks=%d), total_mb=%.2f, latents_stored=%d",
                receiver.total_packets, receiver.json_packets, receiver.ckks_packets,
                receiver.bytes_received / (1024 * 1024),
                await receiver.latent_store.count(),
            )
    except (KeyboardInterrupt, asyncio.CancelledError):
        await receiver.stop()


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
