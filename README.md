# GestureBridge

GestureBridge is a Raspberry Pi + XIAO ESP32S3 edge system for:

- `Translate Mode`
  - sign-to-speech
  - speech-to-sign
- `Learn Mode`
  - teaching stage (target sign shown)
  - practice stage (meaning shown, learner performs sign)

This repository contains a complete runnable software baseline:

- system specification and state machine
- model training/evaluation scripts with INT8 quantization simulation
- translation and learning pipelines
- low-power dual-device coordination simulation
- tests, benchmark scripts, and delivery checklist

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run demo:

```bash
python -m gesturebridge.app
```

Run tests:

```bash
pytest
```

Train baseline model:

```bash
python scripts/train_baseline.py
```

Evaluate model:

```bash
python scripts/evaluate_model.py
```

Power/wakeup simulation:

```bash
python scripts/power_benchmark.py
```

## Project Layout

- `docs/`: architecture/specification/test/delivery docs
- `src/gesturebridge/`: core runtime code
- `scripts/`: training/evaluation/benchmark scripts
- `tests/`: automated checks
