#!/usr/bin/env python3
"""Smoke-test USB camera + ALSA speaker/mic and optionally run GestureBridge pipelines.

Typical Raspberry Pi + Logitech C270 style layout on this machine:
  - Camera: /dev/video0 (MJPEG), microphone often on ALSA card 2 (WEBCAM).
  - USB speaker: ALSA card 3, use plughw for format conversion.

Run from repo root (with venv activated if you use one):

  python scripts/hardware_smoke_test.py
  python scripts/hardware_smoke_test.py --bridge
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def repo_src() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


def ensure_import_path() -> None:
    src = repo_src()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def run(cmd: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(cmd, input=input_bytes, capture_output=True, check=False)


def test_playback(alsa_playback: str) -> None:
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=0.35",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-f",
            "wav",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    play = subprocess.run(
        ["aplay", "-q", "-D", alsa_playback],
        stdin=proc.stdout,
        capture_output=True,
    )
    proc.stdout.close()
    proc.wait(timeout=10)
    if proc.returncode != 0 or play.returncode != 0:
        err = (proc.stderr or b"") + (play.stderr or b"")
        raise RuntimeError(f"playback failed (ffmpeg={proc.returncode}, aplay={play.returncode}): {err.decode()}")


def test_capture_mic(alsa_capture: str, seconds: float) -> None:
    r = run(
        [
            "arecord",
            "-q",
            "-D",
            alsa_capture,
            "-f",
            "cd",
            "-d",
            str(seconds),
            "/tmp/gesturebridge_smoke_mic.wav",
        ]
    )
    if r.returncode != 0:
        raise RuntimeError(f"arecord failed: {r.stderr.decode()}")


def grab_frame_rgb(camera_device: str, width: int, height: int) -> bytes:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "v4l2",
        "-input_format",
        "mjpeg",
        "-video_size",
        f"{width}x{height}",
        "-i",
        camera_device,
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    r = run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg camera failed: {r.stderr.decode()}")
    expected = width * height * 3
    if len(r.stdout) != expected:
        raise RuntimeError(f"frame size mismatch: got {len(r.stdout)} expected {expected}")
    return r.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Hardware smoke test for GestureBridge")
    parser.add_argument("--camera-device", default="/dev/video0", help="V4L2 device path")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--alsa-playback", default="plughw:3,0", help="aplay -D device (USB speaker)")
    parser.add_argument("--alsa-capture", default="plughw:2,0", help="arecord -D device (e.g. webcam mic)")
    parser.add_argument("--skip-mic", action="store_true", help="skip microphone recording test")
    parser.add_argument(
        "--bridge",
        action="store_true",
        help="feed one camera frame through LandmarkExtractor + translate sign→speech",
    )
    args = parser.parse_args()

    print("1) Speaker (short tone)…")
    test_playback(args.alsa_playback)
    print("   OK")

    if not args.skip_mic:
        print("2) Microphone (~1s to /tmp/gesturebridge_smoke_mic.wav)…")
        test_capture_mic(args.alsa_capture, 1.0)
        print("   OK")

    print(f"3) Camera one frame ({args.camera_device}, {args.width}x{args.height})…")
    grab_frame_rgb(args.camera_device, args.width, args.height)
    print("   OK")

    if args.bridge:
        ensure_import_path()
        import numpy as np

        from gesturebridge.bootstrap import build_controller
        from gesturebridge.pipelines.landmarks import LandmarkExtractor

        raw = grab_frame_rgb(args.camera_device, args.width, args.height)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((args.height, args.width, 3))
        extractor = LandmarkExtractor()
        features = extractor.extract(frame)
        controller = build_controller()
        out = controller.run_translate_sign_to_speech(features)
        print("4) GestureBridge translate (sign→speech) from live frame:")
        print(f"   {out}")
        print("   (Project TTS is still a text placeholder; install espeak-ng/piper for real speech.)")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
