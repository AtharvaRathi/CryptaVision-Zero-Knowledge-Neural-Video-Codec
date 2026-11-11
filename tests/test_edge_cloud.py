"""
CryptaVision Phase 6 — Unit & Integration Tests (Edge-to-Cloud TCP Streaming)
=============================================================================
Tests:
1. Binary packet protocol framing (Magic 0x43525950, network byte order, types 0x01 & 0x02).
2. Asynchronous CloudTCPReceiver ingestion and parsing.
3. Stream A JSON routing to the Broadcaster callback.
4. Stream B CKKS ciphertext routing and storage into CloudLatentStore.
5. Incomplete packet and invalid magic number rejection.

Run with:
    pytest tests/test_edge_cloud.py -v
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
from pathlib import Path
import pytest

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cloud.tcp_receiver import (
    CRYPTA_MAGIC_NUMBER,
    HEADER_FORMAT,
    HEADER_SIZE,
    TYPE_JSON,
    TYPE_CKKS,
    CloudLatentRecord,
    CloudLatentStore,
    CloudTCPReceiver,
)


@pytest.mark.asyncio
async def test_cloud_tcp_receiver_json_routing():
    """Verifies that Stream A JSON telemetry is parsed and routed to callback."""
    received_json = []

    def on_json(data):
        received_json.append(data)

    receiver = CloudTCPReceiver(
        host="127.0.0.1",
        port=15555,
        on_json_callback=on_json,
    )
    await receiver.start()

    try:
        # Simulate edge C++ cryptanetworksink connecting and sending packet
        reader, writer = await asyncio.open_connection("127.0.0.1", 15555)

        test_payload = json.dumps({
            "frame_id": 42,
            "num_humans": 1,
            "pose_keypoints": [[[100.0, 200.0]]],
            "clip_embedding": [0.05] * 512,
        }).encode("utf-8")

        # Construct 9-byte header + payload
        header = struct.pack(HEADER_FORMAT, CRYPTA_MAGIC_NUMBER, len(test_payload), TYPE_JSON)
        writer.write(header + test_payload)
        await writer.drain()

        # Wait briefly for async server processing
        await asyncio.sleep(0.1)

        assert len(received_json) == 1
        assert received_json[0]["frame_id"] == 42
        assert received_json[0]["num_humans"] == 1
        assert receiver.json_packets == 1

        writer.close()
        await writer.wait_closed()
    finally:
        await receiver.stop()


@pytest.mark.asyncio
async def test_cloud_tcp_receiver_ckks_routing():
    """Verifies that Stream B CKKS ciphertexts are unpacked and stored in CloudLatentStore."""
    store = CloudLatentStore()
    receiver = CloudTCPReceiver(
        host="127.0.0.1",
        port=15556,
        latent_store=store,
    )
    await receiver.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", 15556)

        dummy_ckks_bytes = b"ENCRYPTED_CKKS_SEAL_VECTOR_PAYLOAD_BYTES"
        envelope = {
            "frame_id": 99,
            "face_idx": 0,
            "box": [50, 60, 150, 160],
            "pts_ns": 3300000000,
        }
        env_bytes = json.dumps(envelope).encode("utf-8")
        env_len_prefix = len(env_bytes).to_bytes(4, byteorder="big")
        payload = env_len_prefix + env_bytes + dummy_ckks_bytes

        header = struct.pack(HEADER_FORMAT, CRYPTA_MAGIC_NUMBER, len(payload), TYPE_CKKS)
        writer.write(header + payload)
        await writer.drain()

        await asyncio.sleep(0.1)

        records = await store.drain()
        assert len(records) == 1
        rec = records[0]
        assert rec.frame_id == 99
        assert rec.face_idx == 0
        assert rec.box == [50, 60, 150, 160]
        assert rec.ckks_bytes == dummy_ckks_bytes
        assert receiver.ckks_packets == 1

        writer.close()
        await writer.wait_closed()
    finally:
        await receiver.stop()


@pytest.mark.asyncio
async def test_invalid_magic_number_rejection():
    """Verifies that packets with invalid magic number are rejected."""
    receiver = CloudTCPReceiver(
        host="127.0.0.1",
        port=15557,
    )
    await receiver.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", 15557)

        # Corrupted magic number (0xDEADBEEF instead of 0x43525950)
        bad_header = struct.pack(HEADER_FORMAT, 0xDEADBEEF, 10, TYPE_JSON)
        writer.write(bad_header + b"0123456789")
        await writer.drain()

        await asyncio.sleep(0.1)
        # Server should disconnect the corrupted client
        assert receiver.total_packets == 0

        writer.close()
        await writer.wait_closed()
    finally:
        await receiver.stop()
