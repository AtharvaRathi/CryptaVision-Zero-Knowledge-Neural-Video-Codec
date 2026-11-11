# CryptaVision

Privacy-preserving edge-to-cloud video analytics with a local sample-video
development workflow.

The edge pipeline splits decoded video into two streams:

- **Stream A:** YOLO pose keypoints and CLIP embeddings sent as JSON metadata
  over a native GStreamer TCP sink, then delivered to the dashboard through SSE.
- **Stream B:** FaceVAE latents encrypted with TenSEAL CKKS for protected
  forensic workflows using HashiCorp Vault.

The project includes C++ GStreamer plugins, a Python edge runner, a FastAPI
backend, a Next.js dashboard, and a forensic reconstruction CLI.

## Local development

On Apple Silicon, use Python 3.11. Install Vault, Node/npm, CMake, pkg-config,
GStreamer and its plugins, GLib, gobject-introspection, and ffmpeg first. See
[the local setup guide](README-local.md) for prerequisites and troubleshooting.

```sh
git clone https://github.com/AtharvaRathi/CryptaVision-Zero-Knowledge-Neural-Video-Codec.git
cd CryptaVision-Zero-Knowledge-Neural-Video-Codec
./download_sample_video.sh
./start_cryptavision.sh --setup
```

After initial setup:

```sh
./start_cryptavision.sh
```

Open the dashboard at <http://localhost:3000> and API documentation at
<http://localhost:8000/docs>. The default sample video loops continuously.
Press Ctrl+C in the launcher's terminal to stop all managed services.

Other input sources remain available:

```sh
./start_cryptavision.sh --source webcam
./start_cryptavision.sh --source /path/to/video.mp4
./start_cryptavision.sh --source rtsp://camera/stream
```

The downloader tries the Wikimedia walking-video URL and falls back to
OpenCV's pedestrian surveillance sample, converting the result to H.264 MP4.
Downloaded videos, model weights, environments and generated build files are
excluded from version control.

## Project layout

| Path | Purpose |
| --- | --- |
| `start.py` | Local service supervisor and setup/preflight checks |
| `edge_node/local_pipeline.py` | Hardware-agnostic edge video runner |
| `pipeline/ai_engine.py` | Pose, CLIP and FaceVAE inference |
| `plugins/` | Native GStreamer frame filter and TCP sink |
| `main.py`, `cloud/` | FastAPI orchestration, SSE and TCP ingest |
| `crypto_engine.py`, `vault_manager.py` | CKKS and Vault integration |
| `frontend/` | Next.js security dashboard |
| `forensic_enclave/` | Forensic reconstruction CLI |
| `tests/` | Unit and integration tests |
| `k8s/`, `helm/`, `Dockerfile` | Deployment scaffolding |

## Validation and current limits

The local workflow has been tested with sample-video playback, native TCP
metadata delivery, dashboard skeleton rendering, video looping and process
cleanup. Relevant test commands are in [README-local.md](README-local.md).

This is a development prototype. Vault dev-mode data is ephemeral. The
existing edge and backend generate separate CKKS contexts, FaceVAE needs
trained weights, and MTCNN needs a CPU fallback for unsupported Apple Silicon
MPS operations. Forensic reconstruction and production deployment have not
been validated end to end. Homomorphic encryption does not by itself provide
a zero-knowledge proof protocol.
