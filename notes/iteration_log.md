# Phase 2 Iteration Log

Append-only log of every iteration in the Phase 2 word-recognition sprint.
Each entry follows: **What was attempted / What works / What broke / Files
touched / Numbers / Sources & URLs**.

Plan reference: `~/.claude/plans/i-have-seen-claude-nifty-brooks.md`.
Branch: `shufeng`. Active mode: iteration-driven, ship-first.

---

## Iteration 1 — 2026-05-05 — Speech→sign upgrade for 5 known words

### What was attempted

The user reported the loudest pain point: when they say "electrical" the
speech-to-sign output is a 10-image array of letter gestures. Real ASL
has dedicated word-level signs. IT-1 fixes this for a small starter
vocabulary (`hello`, `thanks/thank you`, `yes`, `no`, `help`) before
training anything — proves the architecture for video-based
speech-to-sign output.

### What works

- Backend `MainRuntime.run_speech_to_sign` now does per-token lookup. If
  a token has a video clip on disk under `assets/word_clips/`, the
  response uses the clip filename (e.g. `hello.mp4`); otherwise it
  falls back to letter-spelling that token.
- Aliases handled in `WORD_CLIP_MAP` (`thanks` and `thank` both map to
  `thank_you.mp4`; `hi`/`hey` map to `hello.mp4`; `yeah`/`yep` map to
  `yes.mp4`; `nope` maps to `no.mp4`).
- `letters` and `sign_assets` lists are now strictly aligned 1:1 — the
  UI can map index-to-index without surprises.
- Web server serves `/assets/word_clips/<name>.mp4` with `video/mp4`
  MIME type, mirroring the existing `/assets/signs/` path. Missing clips
  return 404 cleanly.
- Web UI `renderSignGallery` detects `.mp4`/`.webm` extensions and
  emits a `<video autoplay loop muted playsinline>` element instead of
  `<img>`. Letter results unchanged.
- Smoke tests on Mac (Python 3.12 venv):
  - `'hello'` → `letters=['HELLO']`, `sign_assets=['hello.mp4']`
  - `'thanks'` → `letters=['THANKS']`, `sign_assets=['thank_you.mp4']`
  - `'electrical'` → letter array (fallback proven, no clip exists)
  - `'hello electrical'` → mixed `['hello.mp4', 'E.jpg', 'L.jpg', …]`
  - HTTP `GET /assets/word_clips/hello.mp4` → `200`, `video/mp4`, 85891 bytes
  - HTTP `GET /assets/word_clips/missing.mp4` → `404`
  - HTTP `GET /assets/signs/A.jpg` → `200`, `image/jpeg`, 12635 bytes

### What broke

- First attempt with Python 3.9 venv failed (`dataclass(slots=True)` is
  3.10+). Switched to `/opt/homebrew/bin/python3.12`.
- `aslbricks.org` returns 406 to the default curl User-Agent. Fixed by
  passing a Chrome UA via `-A`.
- `aslbricks.org` does not have `hello.mp4`. Fell back to
  `media.signbsl.com` (Start ASL clip), reached via `signasl.org`.
- Python emitted a `SyntaxWarning: invalid escape sequence '\.'` in the
  inline JS regex. Replaced regex with `endsWith('.mp4')`.

### Files touched

- `src/gesturebridge/system/main_runtime.py:67-87` — added `WORD_CLIP_MAP`
- `src/gesturebridge/system/main_runtime.py:412-441` — rewrote
  `run_speech_to_sign` to do per-token word→clip lookup with letter
  fallback, kept `letters` and `sign_assets` aligned 1:1
- `src/gesturebridge/config.py:135-144` — added `WebUIConfig.word_clips_dir`
- `src/gesturebridge/ui/web.py:794-832` — extended `/assets/...` route
  to also serve `/assets/word_clips/`, added MP4/WebM MIME types
- `src/gesturebridge/ui/web.py:511-540` — `renderSignGallery` branches
  to `<video>` for video assets
- `assets/word_clips/SOURCES.md` — new, documents clip provenance
- `scripts/fetch_word_clips.sh` — new, reproduces the IT-1 download
- `.gitignore` — `assets/word_clips/*.mp4`, `external/WLASL/`
- `notes/iteration_log.md` — this entry
- `notes/yizheng_wechat.md` — first Chinese sync

### Numbers

- 5 / 5 target words have working clips on disk (durations 2.2-2.7 s,
  resolutions 640×360 to 1920×1080).
- Backend logic verified against 8 utterances; no regression in
  letter-spelling fallback.
- HTTP smoke test: 200 / 404 paths both correct.
- No new runtime dependencies (still numpy + opencv-python-headless on
  dev; Pi runtime unchanged).

### Sources / URLs (for the user to verify)

- WLASL v0.3 JSON (canonical gloss list, used to discover URLs):
  `https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json`
- `hello.mp4`: `https://media.signbsl.com/videos/asl/startasl/mp4/hello.mp4`
  (referrer `https://www.signasl.org/`).
- `help.mp4`, `no.mp4`, `yes.mp4`: `http://aslbricks.org/New/ASL-Videos/{word}.mp4`
- `thank_you.mp4`: `http://aslbricks.org/New/ASL-Videos/thank%20you.mp4`
- yt-dlp installed `--user` for IT-2 (data acquisition step).

### Acceptance check

User runs the app on Mac; speech-to-sign tab; says or types `hello` →
UI plays the HELLO clip in a `<video>` element. Says `electrical` →
letter sequence (fallback). Letter mode and Read mode are untouched.

### Next iteration entry point

IT-2: download WLASL-100 video set + build `(N, 30, 63)` landmark
tensor for training.

---

