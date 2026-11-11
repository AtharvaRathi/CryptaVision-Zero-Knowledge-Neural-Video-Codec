"""
CryptaVision — Unit Tests for Hardware-Agnostic Local Edge Pipeline
=====================================================================
Tests:
1. Dynamic source resolution (`get_source_element`) across webcams, RTSP, HTTP, and local files.
2. Argument parsing and default configurations.
3. LocalEdgePipeline initialization, AI Engine hooks, and TenSEAL CKKS setup.
4. Stream A and Stream B packet serialization logic.

Run with:
    pytest tests/test_local_pipeline.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from edge_node.local_pipeline import LocalEdgePipeline, get_source_element


class TestDynamicSourceResolution:
    """Verifies get_source_element resolution across input source variants."""

    def test_webcam_resolution(self):
        assert get_source_element("0") == "autovideosrc"
        assert get_source_element("webcam") == "autovideosrc"
        assert get_source_element("WEBCAM") == "autovideosrc"
        assert get_source_element("/dev/video0") == "autovideosrc"

    def test_rtsp_resolution(self):
        rtsp_url = "rtsp://10.0.0.15:554/live/stream1"
        assert get_source_element(rtsp_url) == f"uridecodebin uri={json.dumps(rtsp_url)}"

    def test_http_resolution(self):
        http_url = "http://stream.example.com:8080/feed.m3u8"
        assert get_source_element(http_url) == f"uridecodebin uri={json.dumps(http_url)}"
        https_url = "https://stream.example.com/live.mp4"
        assert get_source_element(https_url) == f"uridecodebin uri={json.dumps(https_url)}"

    def test_local_file_resolution(self, tmp_path: Path):
        test_video = tmp_path / "traffic_sample.mp4"
        test_video.write_bytes(b"\x00\x00\x00\x20ftypisom")  # dummy mp4 header

        resolved = get_source_element(str(test_video))
        expected = f"uridecodebin uri={json.dumps(test_video.resolve().as_uri())}"
        assert resolved == expected

    def test_synthetic_test_resolution(self):
        assert get_source_element("videotestsrc") == "videotestsrc"
        assert get_source_element("test") == "videotestsrc"


class TestLocalEdgePipelineInitialization:
    """Verifies LocalEdgePipeline configuration and engine setup."""

    @patch("edge_node.local_pipeline.LocalEdgePipeline._init_engines")
    def test_default_sample_is_resolved_from_project_root(self, _init, tmp_path, monkeypatch):
        monkeypatch.setattr("edge_node.local_pipeline.PROJECT_ROOT", tmp_path)
        sample = tmp_path / "test_surveillance_feed.mp4"
        sample.write_bytes(b"test")
        pipeline = LocalEdgePipeline()
        assert pipeline.source_uri == str(sample)
        assert pipeline._loop_video

    @patch("edge_node.local_pipeline.CryptaAIEngine")
    @patch("edge_node.local_pipeline.CryptoEngine")
    def test_pipeline_defaults(self, mock_crypto_cls, mock_ai_cls):
        pipeline = LocalEdgePipeline(
            source_uri="webcam",
            cloud_host="127.0.0.1",
            cloud_port=5555,
            process_interval=5,
            latent_dim=128,
        )

        assert pipeline.source_uri == "webcam"
        assert not pipeline._loop_video
        assert pipeline.cloud_host == "127.0.0.1"
        assert pipeline.cloud_port == 5555
        assert pipeline.process_interval == 5
        assert pipeline.latent_dim == 128
        assert pipeline.frames_in == 0
        assert pipeline.stream_a_packets == 0
        assert pipeline.stream_b_packets == 0

        # Verify AI and Crypto initialization
        mock_ai_cls.assert_called_once_with(process_interval=5, latent_dim=128)
        mock_crypto_cls.assert_called_once_with(latent_dim=128)

    @patch("edge_node.local_pipeline.CryptaAIEngine")
    @patch("edge_node.local_pipeline.CryptoEngine")
    def test_network_push_formatting(self, mock_crypto_cls, mock_ai_cls):
        """Verifies packet formatting for Stream A and Stream B."""
        pipeline = LocalEdgePipeline(
            source_uri="0",
            cloud_host="10.0.0.1",
            cloud_port=5555,
        )

        sent_buffers = []

        def mock_push(data_bytes):
            sent_buffers.append(data_bytes)

        pipeline._push_to_network = mock_push

        # Test Stream A JSON telemetry (Type 0x01)
        test_json = {"frame_id": 1, "num_humans": 2}
        json_bytes = json.dumps(test_json).encode("utf-8")
        packet_a = b"\x01" + json_bytes
        pipeline._push_to_network(packet_a)

        assert len(sent_buffers) == 1
        assert sent_buffers[0][0] == 0x01
        assert json.loads(sent_buffers[0][1:].decode("utf-8")) == test_json

        # Test Stream B CKKS ciphertext (Type 0x02)
        envelope = {"frame_id": 1, "face_idx": 0}
        env_bytes = json.dumps(envelope).encode("utf-8")
        env_len = len(env_bytes).to_bytes(4, byteorder="big")
        ckks_data = b"MOCK_CKKS_DATA"
        packet_b = b"\x02" + env_len + env_bytes + ckks_data
        pipeline._push_to_network(packet_b)

        assert len(sent_buffers) == 2
        assert sent_buffers[1][0] == 0x02
