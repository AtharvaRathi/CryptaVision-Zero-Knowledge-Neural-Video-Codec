#!/bin/sh
# Resolve paths independently of the caller's working directory.
set -eu
CRYPTAVISION_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -x "$CRYPTAVISION_ROOT/venv/bin/python" ]; then
    exec "$CRYPTAVISION_ROOT/venv/bin/python" "$CRYPTAVISION_ROOT/start.py" "$@"
fi
if [ -x /opt/homebrew/bin/python3.11 ]; then
    exec /opt/homebrew/bin/python3.11 "$CRYPTAVISION_ROOT/start.py" "$@"
fi
exec python3.11 "$CRYPTAVISION_ROOT/start.py" "$@"
