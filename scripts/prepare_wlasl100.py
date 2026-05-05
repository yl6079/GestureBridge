#!/usr/bin/env python3
"""Download WLASL-100 clips for Phase 2 dynamic-gesture training.

Strategy:
- Read the official WLASL_v0.3.json (cached at --json or fetched from GitHub).
- Pick the top-N glosses by instance count (default 100 = WLASL-100 proxy).
- For each instance, try direct-MP4 sources first (aslbricks, aslsignbank,
  signingsavvy, aslsearch, etc.). Fall back to yt-dlp on YouTube if none of
  the direct sources work.
- Save into data/wlasl100/videos/<gloss>/<video_id>.mp4 and write a manifest
  CSV mapping clip → gloss → split. Idempotent: existing files are skipped.

Usage:
    python scripts/prepare_wlasl100.py --top-n 100 --max-per-gloss 30
    python scripts/prepare_wlasl100.py --download-only-direct  # skip yt-dlp

License: WLASL is C-UDA. Downloaded videos are gitignored. Do NOT redistribute.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import ssl
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


# aslsignbank.haskins.yale.edu serves a wrong-hostname cert; we accept it
# only because we already trust the WLASL JSON URL pointing here.
_INSECURE_SSL_CTX = ssl.create_default_context()
_INSECURE_SSL_CTX.check_hostname = False
_INSECURE_SSL_CTX.verify_mode = ssl.CERT_NONE


def _referer_for(url: str) -> str:
    lower = url.lower()
    if "signingsavvy.com" in lower:
        return "https://www.signingsavvy.com/"
    if "aslsignbank.haskins.yale.edu" in lower:
        return "https://aslsignbank.haskins.yale.edu/"
    if "aslbricks.org" in lower:
        return "http://aslbricks.org/"
    if "signbsl.com" in lower:
        return "https://www.signasl.org/"
    return "https://www.google.com/"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Sources we know yield direct MP4s. Order matters: cheapest/fastest first.
DIRECT_SOURCES = {
    "aslbrick", "aslbricks",
    "aslsignbank",
    "signingsavvy",
    "aslsearch",
    "asldeafined",
    "startasl",
    "elementalasl",
    "lillybauer",
    "nabboud",
    "valencia-asl",
    "scott",
}


def fetch_json(path: Path | None) -> list[dict]:
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    url = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
    print(f"[prep] fetching {url}")
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=60) as r:
        data = r.read()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return json.loads(data.decode("utf-8"))


def select_top_n(entries: list[dict], n: int) -> list[dict]:
    sized = sorted(entries, key=lambda e: -len(e.get("instances", [])))
    return sized[:n]


def is_youtube(url: str) -> bool:
    return "youtube.com/" in url or "youtu.be/" in url


def is_direct_video_url(url: str) -> bool:
    lower = url.lower()
    return any(lower.endswith(ext) for ext in (".mp4", ".webm", ".mov"))


def try_direct_download(url: str, dest: Path, timeout: int = 30) -> bool:
    """Single shot direct MP4 fetch with browser UA + per-host Referer."""
    try:
        headers = {
            "User-Agent": UA,
            "Referer": _referer_for(url),
            "Accept": "video/mp4,video/*;q=0.9,*/*;q=0.5",
        }
        req = Request(url, headers=headers)
        # aslsignbank presents a cert for the wrong hostname; we still trust the
        # original WLASL JSON URL list, so opt into a relaxed SSL context for
        # only that host.
        ctx = _INSECURE_SSL_CTX if "aslsignbank.haskins.yale.edu" in url.lower() else None
        with urlopen(req, timeout=timeout, context=ctx) as r:
            ctype = r.headers.get("Content-Type", "")
            if "video" not in ctype.lower() and not is_direct_video_url(url):
                # If we can't tell from content-type or extension, accept anyway —
                # Some servers return application/octet-stream for MP4.
                pass
            tmp = dest.with_suffix(".part")
            with tmp.open("wb") as f:
                shutil.copyfileobj(r, f, length=64 * 1024)
            if tmp.stat().st_size < 8 * 1024:
                tmp.unlink(missing_ok=True)
                return False
            tmp.rename(dest)
            return True
    except Exception as exc:
        # Most failures are 404/403/timeout; suppress noise but log root cause briefly.
        print(f"  [direct-fail] {url[:60]}... -> {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def try_ytdlp_download(url: str, dest: Path, timeout: int = 90) -> bool:
    """yt-dlp fallback. Slow and YouTube can 403; we accept some failure rate."""
    if shutil.which("yt-dlp") is None and shutil.which("python3") is not None:
        cmd = ["python3", "-m", "yt_dlp"]
    else:
        cmd = ["yt-dlp"]
    cmd += [
        "--quiet",
        "--no-warnings",
        "-f", "best[ext=mp4]/best",
        "-o", str(dest),
        "--socket-timeout", "20",
        url,
    ]
    try:
        subprocess.run(cmd, timeout=timeout, check=True)
        return dest.exists() and dest.stat().st_size > 8 * 1024
    except Exception as exc:
        print(f"  [yt-fail] {url[:60]}... -> {type(exc).__name__}", file=sys.stderr)
        # Clean up partial files
        for p in dest.parent.glob(dest.name + "*"):
            p.unlink(missing_ok=True)
        return False


def download_instance(
    gloss: str,
    inst: dict,
    out_dir: Path,
    *,
    skip_youtube: bool,
) -> tuple[bool, str]:
    """Try every URL associated with this instance until one works."""
    video_id = str(inst.get("video_id", "unknown"))
    src = str(inst.get("source", "")).lower()
    url = str(inst.get("url", ""))
    dest = out_dir / f"{video_id}.mp4"
    if dest.exists():
        return True, f"cached {src}"
    if not url:
        return False, "no_url"
    if is_youtube(url):
        if skip_youtube:
            return False, "skipped_youtube"
        ok = try_ytdlp_download(url, dest)
        return ok, ("ytdlp_ok" if ok else "ytdlp_fail")
    # Treat anything non-YouTube as direct MP4 attempt.
    ok = try_direct_download(url, dest)
    return ok, ("direct_ok" if ok else "direct_fail")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=Path("/tmp/WLASL_v0.3.json"),
                   help="cached WLASL_v0.3.json (downloaded if missing)")
    p.add_argument("--out", type=Path, default=Path("data/wlasl100"),
                   help="output dir")
    p.add_argument("--top-n", type=int, default=100,
                   help="number of glosses to use (top by instance count)")
    p.add_argument("--max-per-gloss", type=int, default=30,
                   help="cap on instances downloaded per gloss")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel download workers")
    p.add_argument("--download-only-direct", action="store_true",
                   help="skip YouTube/yt-dlp fallback (fast, partial dataset)")
    p.add_argument("--manifest", type=Path, default=None,
                   help="manifest CSV path (default: <out>/manifest.csv)")
    args = p.parse_args()

    out_root = args.out
    videos_root = out_root / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (out_root / "manifest.csv")
    labels_path = out_root / "labels.txt"

    entries = fetch_json(args.json)
    print(f"[prep] WLASL has {len(entries)} glosses; selecting top {args.top_n}")
    top = select_top_n(entries, args.top_n)

    labels_path.write_text("\n".join(e["gloss"] for e in top) + "\n", encoding="utf-8")
    print(f"[prep] wrote labels: {labels_path}")

    # Build the work list: one row per instance up to max_per_gloss.
    work: list[tuple[str, dict]] = []
    for entry in top:
        gloss = entry["gloss"]
        gloss_dir = videos_root / gloss.replace(" ", "_")
        gloss_dir.mkdir(parents=True, exist_ok=True)
        # Sort instances so direct sources come first (faster to give up on bad rows).
        insts = sorted(
            entry.get("instances", []),
            key=lambda i: (
                0 if str(i.get("source", "")).lower() in DIRECT_SOURCES else 1,
                0 if not is_youtube(str(i.get("url", ""))) else 1,
            ),
        )
        for inst in insts[: args.max_per_gloss]:
            work.append((gloss, inst))

    print(f"[prep] {len(work)} download tasks queued; workers={args.workers}")
    rows: list[dict] = []
    ok_count = 0
    fail_count = 0
    skipped_count = 0

    def task(args_pair):
        gloss, inst = args_pair
        gloss_slug = gloss.replace(" ", "_")
        gloss_dir = videos_root / gloss_slug
        ok, msg = download_instance(
            gloss, inst, gloss_dir, skip_youtube=args.download_only_direct
        )
        return gloss, gloss_slug, inst, ok, msg

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(task, w) for w in work]
        for i, fut in enumerate(as_completed(futures), 1):
            gloss, gloss_slug, inst, ok, msg = fut.result()
            video_id = str(inst.get("video_id", "?"))
            rows.append({
                "gloss": gloss,
                "gloss_slug": gloss_slug,
                "video_id": video_id,
                "split": inst.get("split", "train"),
                "source": inst.get("source", ""),
                "url": inst.get("url", ""),
                "frame_start": inst.get("frame_start", 1),
                "frame_end": inst.get("frame_end", -1),
                "ok": ok,
                "result": msg,
                "path": str(videos_root / gloss_slug / f"{video_id}.mp4") if ok else "",
            })
            if ok:
                ok_count += 1
            elif "skipped" in msg:
                skipped_count += 1
            else:
                fail_count += 1
            if i % 20 == 0 or i == len(futures):
                print(f"[prep] progress {i}/{len(futures)} ok={ok_count} fail={fail_count} skip={skipped_count}")

    # Write manifest.
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[prep] manifest: {manifest_path} ({len(rows)} rows)")

    # Per-class coverage summary.
    by_gloss: dict[str, list[dict]] = {}
    for r in rows:
        by_gloss.setdefault(r["gloss"], []).append(r)
    covered = [g for g, rs in by_gloss.items() if any(r["ok"] for r in rs)]
    print(f"[prep] glosses with ≥1 clip: {len(covered)}/{len(by_gloss)}")
    print(f"[prep] glosses with ≥3 clips:")
    rich = [(g, sum(r['ok'] for r in rs)) for g, rs in by_gloss.items()]
    rich.sort(key=lambda x: -x[1])
    print(f"  top: {rich[:5]}")
    print(f"  bottom: {rich[-5:]}")


if __name__ == "__main__":
    main()
