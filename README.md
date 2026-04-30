# GestureBridge (Test Workspace)

This `test` directory is the runnable and trainable workspace for GestureBridge. It is used for:

- ASL29 dataset preparation, training, evaluation, and export
- Raspberry Pi-side TFLite inference and benchmarking
- Runtime health checks, serial monitoring, and baseline automated tests

---

## 1. Directory Layout

- `assets/`: test/sample resources (including sign image samples)
- `artifacts/`: generated outputs (models, logs, exports)
- `src/gesturebridge/`: core application code
- `scripts/`: training, evaluation, export, realtime inference, and diagnostics
- `tests/`: pytest-based automated tests
- `deploy/`: deployment-related assets

---

## 2. Environment Setup

Use a Python 3.11 virtual environment inside the `test` directory.

```bash
python3 -m venv .venv311
source .venv311/bin/activate
pip install -e ".[dev,ml]"
```

If you already use `uv` in this directory, you can continue using the existing `.uv-python` and `.venv311`.

---

## 3. Common Commands

### 3.1 Run the app

```bash
python -m gesturebridge.app
```

### 3.2 Run tests

```bash
pytest
```

### 3.3 Runtime health check

```bash
python scripts/healthcheck_runtime.py
```

### 3.4 Serial monitor (when external devices are connected)

```bash
python scripts/serial_monitor.py
```

---

## 4. ASL29 Training and Inference Pipeline

ASL29 classes: `A-Z + del + nothing + space` (29 classes total).

Standard pipeline:

```bash
# 1) Prepare dataset
python scripts/prepare_asl29.py --input-dir <your_dataset_dir>

# 2) Train
python scripts/train_mobilenetv3_asl29.py

# 3) Evaluate
python scripts/evaluate_mobilenetv3_asl29.py

# 4) Export TFLite INT8
python scripts/export_tflite_int8_asl29.py

# 5) Benchmark on Raspberry Pi
python scripts/benchmark_tflite_rpi.py

# 6) Realtime inference (GUI)
python scripts/run_realtime_asl29.py

# 7) Realtime inference (headless)
python scripts/run_realtime_asl29_headless.py
```

---

## 5. Test Coverage

Current test coverage in `tests/`:

- `test_translate_mode.py`: translate mode behavior
- `test_learn_mode.py`: learn mode behavior
- `test_state_machine.py`: state machine logic
- `test_training_pipeline.py`: key training pipeline checks
- `test_daemon_serial.py`: daemon + serial communication flow

Run a subset by keyword:

```bash
pytest -k "translate or learn"
```

---

## 6. Maintenance Notes

- Keep documentation consolidated in this `README.md`
- Update command examples when script behavior changes
- Update the "Directory Layout" section when new folders are added
