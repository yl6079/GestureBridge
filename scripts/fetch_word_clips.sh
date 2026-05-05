#!/usr/bin/env bash
# Fetch the 5 IT-1 reference word clips into assets/word_clips/.
# Sources documented in assets/word_clips/SOURCES.md.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p assets/word_clips
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

# hello: Start ASL clip mirrored on signbsl, reached via signasl.org
curl -sSL -A "$UA" -e "https://www.signasl.org/" \
  -o assets/word_clips/hello.mp4 \
  "https://media.signbsl.com/videos/asl/startasl/mp4/hello.mp4"

# help, no, yes, thank_you: aslbricks.org direct MP4s
for w in help no yes; do
  curl -sSL -A "$UA" \
    -o "assets/word_clips/${w}.mp4" \
    "http://aslbricks.org/New/ASL-Videos/${w}.mp4"
done
curl -sSL -A "$UA" \
  -o "assets/word_clips/thank_you.mp4" \
  "http://aslbricks.org/New/ASL-Videos/thank%20you.mp4"

echo "Downloaded:"
ls -lh assets/word_clips/*.mp4
