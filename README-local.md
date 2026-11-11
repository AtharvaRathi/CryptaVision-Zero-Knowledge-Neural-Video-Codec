# Run CryptaVision locally

From the project directory, start all four services with:

```sh
./start_cryptavision.sh
```

The launcher selects the project's Python 3.11 `venv` automatically; shell
activation is optional. The default input is `test_surveillance_feed.mp4` in
the project root. Download it once before starting:

```sh
./download_sample_video.sh
./start_cryptavision.sh
```

The downloader requires `curl` and `ffmpeg`. It tries the requested Wikimedia
walking clip, then falls back to OpenCV's public pedestrian surveillance
[vtest.avi](https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi)
if Wikimedia is unavailable. It converts the video to H.264 MP4 with no audio.
The fallback is 768×576; it is not a 1080p source. Existing sample files are reused.

Local files repeat automatically using
[GStreamer segment seeking](https://gstreamer.freedesktop.org/documentation/additional/design/seeking.html)
at real-time playback speed. Each repeat retains the pipeline and TCP connection.
The same default applies to `venv/bin/python edge_node/local_pipeline.py`, even
when invoked from another directory. Other inputs, including the live webcam,
remain available:

```sh
./start_cryptavision.sh --source sample.mp4
./start_cryptavision.sh --source webcam
./start_cryptavision.sh --source rtsp://camera/stream
./start_cryptavision.sh --source test
```

Open [the dashboard](http://localhost:3000) or
[the API docs](http://localhost:8000/docs). Logs carry service labels and use
colors in a terminal (`NO_COLOR=1` disables colors). Press Ctrl+C once to stop
Vault, FastAPI, Next.js and the edge process, including their child processes.
Shutdown allows eight seconds for cleanup before killing remaining process
groups and reaping direct children. Startup failures and service exits also
trigger cleanup. Existing processes on the required ports are never killed.

The dashboard reports `LIVE` only while semantic frames are arriving. An SSE
connection without metadata shows `WAITING FOR FRAMES`; a feed with no new
frames for 15 seconds shows `FEED STALLED`. A valid frame with zero people
shows `NO PEOPLE DETECTED IN CURRENT FRAME` while its frame counter and CLIP
embeddings continue updating.

For a fresh checkout, install Homebrew Python 3.11, Vault, Node/npm, CMake,
pkg-config, GStreamer (including base/good/bad/libav plugins), GLib and
gobject-introspection first. Then run:

```sh
./start_cryptavision.sh --setup
```

`--setup` creates `venv` with Python 3.11 if needed, installs all four Python
requirements files plus PyGObject, builds both C++ plugins using CMake, runs
`npm ci`, and launches the stack. TenSEAL must install from a binary wheel to
avoid the unsupported source-build path. On Apple Silicon the preferred
interpreter is `/opt/homebrew/bin/python3.11`. The native plugin outputs are
`plugins/build/crypta_frame_filter.dylib` and
`plugins/build/crypta_network_sink.dylib`, matching the existing CMake naming.
On systems where PyGObject requires compilation, its native development
dependencies must be installed before setup.

```sh
./start_cryptavision.sh --check
./start_cryptavision.sh --setup --check
```

These commands validate imports, GStreamer plugin discovery and free ports
without starting the services. Python users can also invoke `python3.11 start.py`.
Service readiness defaults to 120 seconds; adjust with `--timeout 180`.

The launcher sets `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib:…`,
`GST_PLUGIN_PATH=<project>/plugins/build:…` and the virtual environment for
every child. It waits for Vault, enables `cryptavision/` as KV v2, and passes
`VAULT_MOUNT_POINT=cryptavision` to the backend. It requires a connected Vault
and TCP ingest on 5555 before launching the dashboard and edge. API and
dashboard ports are fixed at 8000 and 3000; Vault uses 8200 and 8201.

Vault runs in development mode with ephemeral in-memory data and a fresh root
token on each launch. The token is passed through the environment, redacted
from logs, and never written to `~/.vault-token`. Stopping Vault discards its
keys. This launcher is for local development. First-run model loading may
download YOLO/CLIP weights, and macOS may request webcam access.

The startup work does not validate forensic reconstruction: the existing edge
and backend generate separate CKKS contexts, and FaceVAE uses random weights
unless trained weights are provided. Those existing model/key provisioning
limitations need resolution before a forensic end-to-end validation.
The synthetic smoke test also showed MTCNN warnings for unsupported MPS
adaptive pooling; face detection on Apple Silicon needs a CPU fallback or a
compatible detector before validating Stream B.

Supervisor regression tests (no camera/model downloads):

```sh
venv/bin/python -m pytest tests/test_start.py -q
```

The native video-loop/TCP regression test requires GStreamer, the built
plugins, ffmpeg and loopback networking:

```sh
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib GST_PLUGIN_PATH="$PWD/plugins/build" \
  venv/bin/python -m pytest tests/test_video_loop.py -q
```
