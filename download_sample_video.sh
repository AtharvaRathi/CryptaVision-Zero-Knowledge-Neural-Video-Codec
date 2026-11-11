#!/bin/sh
# Fetch a pedestrian test clip and convert it to a genuine H.264 MP4.
set -eu
CRYPTAVISION_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT="$CRYPTAVISION_ROOT/test_surveillance_feed.mp4"
command -v curl >/dev/null
command -v ffmpeg >/dev/null || { echo 'Install ffmpeg first (brew install ffmpeg).' >&2; exit 1; }
if [ -s "$OUTPUT" ]; then
    echo "Sample already exists: $OUTPUT"
    exit 0
fi
WORK=$(mktemp -d "$CRYPTAVISION_ROOT/.sample-download.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
PRIMARY='https://upload.wikimedia.org/wikipedia/commons/transcoded/8/87/Woman_walking_in_a_city_street_-_1080p.webm/Woman_walking_in_a_city_street_-_1080p.webm.480p.vp9.webm'
FALLBACK='https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi'
if ! curl --fail --location --retry 2 --connect-timeout 15 --max-time 180 "$PRIMARY" -o "$WORK/source"; then
    echo 'Wikimedia clip unavailable; downloading OpenCV pedestrian surveillance sample.'
    curl --fail --location --retry 2 --connect-timeout 15 --max-time 180 "$FALLBACK" -o "$WORK/source"
fi
ffmpeg -hide_banner -loglevel warning -nostdin -i "$WORK/source" -map 0:v:0 \
    -an -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p -movflags +faststart "$WORK/sample.mp4"
mv "$WORK/sample.mp4" "$OUTPUT"
echo "Ready: $OUTPUT"
