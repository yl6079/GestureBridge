# Release Candidate Notes (v0.1.0)

## Implemented Scope

- Translate mode:
  - Sign-to-speech path with confidence gating.
  - Speech-to-sign path with offline ASR abstraction and reference image key output.
- Learn mode:
  - Teaching stage and practice stage with immediate correctness feedback.
  - Session stats (`attempts`, `accuracy`, streak tracking).
- Dual-device coordination:
  - XIAO activity detection and Raspberry Pi wake/sleep lifecycle.
  - Inactivity-based cooldown and sleep transition.

## Validation Results

- Baseline training/evaluation scripts executed successfully.
- Quantized classifier evaluation report generated in `artifacts/evaluation_report.json`.
- Power/wakeup behavior report generated in `artifacts/power_report.json`.
- Automated tests pass (`5 passed`).

## Remaining Integration Work for Real Hardware

- Replace simulated landmark extractor with full MediaPipe runtime.
- Replace ASR/TTS placeholders with selected on-device engines.
- Add camera/microphone/display hardware drivers and UI surface.
- Run target-device profiling for FPS, RAM peak, and battery/power figures.
