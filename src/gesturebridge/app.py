from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from threading import Thread
import subprocess
import sys

import numpy as np

from gesturebridge.bootstrap import build_controller
from gesturebridge.config import SystemConfig
from gesturebridge.devices.xiao import parse_serial_line
from gesturebridge.system.daemon import StandbyDaemon

def run_demo() -> None:
    controller = build_controller()
    print("Wake:", controller.wake_if_needed(activity_level=0.9))

    sample_frame = np.random.default_rng(42).normal(size=(21, 3)).astype(np.float32)
    print("Translate sign->speech:", controller.run_translate_sign_to_speech(sample_frame))
    print("Translate speech->sign:", controller.run_translate_speech_to_sign("hello"))

    print("Learn teaching:", controller.run_learn_teaching(sample_frame, target_sign_id=0))
    print("Learn practice:", controller.run_learn_practice(sample_frame, target_sign_id=1))
    print("Housekeeping:", controller.housekeeping())


def _open_local_ui(url: str, kiosk_mode: bool) -> None:
    if kiosk_mode:
        for browser_cmd in ("chromium-browser", "chromium"):
            browser_bin = shutil.which(browser_cmd)
            if browser_bin:
                subprocess.Popen([browser_bin, "--kiosk", "--noerrdialogs", "--disable-infobars", url])
                print(f"Opened kiosk browser: {browser_cmd}")
                return
    xdg_open = shutil.which("xdg-open")
    if xdg_open:
        subprocess.Popen([xdg_open, url])
        print("Opened browser via xdg-open")
        return
    print(f"Browser auto-open skipped. Open manually: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GestureBridge runtime entrypoint")
    parser.add_argument("--demo", action="store_true", help="run quick pipeline demo")
    parser.add_argument("--benchmark-asl29", action="store_true", help="benchmark TFLite ASL29 runtime")
    parser.add_argument("--run-main", action="store_true", help="run main interaction process")
    parser.add_argument("--run-daemon", action="store_true", help="run standby daemon process")
    parser.add_argument("--mock-serial", default="", help="comma-separated mock serial events for daemon")
    parser.add_argument("--speech", default="", help="speech input for speech->sign mode")
    args = parser.parse_args()
    cfg = SystemConfig()
    if sys.version_info >= (3, 13):
        venv311_python = Path.cwd() / ".venv311" / "bin" / "python"
        if venv311_python.exists():
            completed = subprocess.run([str(venv311_python), "-m", "gesturebridge.app", *sys.argv[1:]], check=False)
            raise SystemExit(completed.returncode)
        raise RuntimeError(
            "Python 3.13 is not supported for this TensorFlow stack. "
            "Use .venv311: source .venv311/bin/activate"
        )

    if args.demo:
        run_demo()
        return

    if args.benchmark_asl29:
        from gesturebridge.pipelines.asl29_tflite import ASL29TFLiteRuntime

        runtime = ASL29TFLiteRuntime(
            model_path=Path(cfg.asl29.export.fp32_tflite_path),
            labels_path=Path(cfg.asl29.data.labels_path),
            threads=cfg.asl29.runtime.tflite_threads,
            image_size=cfg.asl29.data.image_size,
            top_k=cfg.asl29.runtime.preview_top_k,
        )
        rng = np.random.default_rng(123)
        elapsed: list[float] = []
        for _ in range(cfg.asl29.runtime.benchmark_warmup + cfg.asl29.runtime.benchmark_iterations):
            frame = rng.integers(0, 255, size=(cfg.asl29.runtime.webcam_height, cfg.asl29.runtime.webcam_width, 3), dtype=np.uint8)
            result = runtime.predict(frame)
            elapsed.append(result.latency_ms)
        warmup = cfg.asl29.runtime.benchmark_warmup
        effective = elapsed[warmup:]
        report = {
            "iterations": len(effective),
            "threads": cfg.asl29.runtime.tflite_threads,
            "avg_latency_ms": float(np.mean(effective)),
            "p95_latency_ms": float(np.percentile(effective, 95)),
            "estimated_fps": float(1000.0 / max(np.mean(effective), 1e-6)),
        }
        out = Path(cfg.asl29.runtime.benchmark_report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"saved={out}")
        return

    if args.run_main:
        from gesturebridge.pipelines.asl29_tflite import ASL29TFLiteRuntime
        from gesturebridge.pipelines.asr import OfflineASR
        from gesturebridge.pipelines.tts import TTSOutput
        from gesturebridge.system.main_runtime import MainRuntime
        from gesturebridge.ui.web import UIState, build_web_server

        infer = ASL29TFLiteRuntime(
            model_path=Path(cfg.asl29.export.fp32_tflite_path),
            labels_path=Path(cfg.asl29.data.labels_path),
            threads=cfg.asl29.runtime.tflite_threads,
            image_size=cfg.asl29.data.image_size,
            top_k=cfg.asl29.runtime.preview_top_k,
            use_hand_crop=cfg.asl29.runtime.use_hand_crop,
            hand_cropper_model_path=Path(cfg.asl29.runtime.hand_landmarker_path),
        )
        main_runtime = MainRuntime(config=cfg, infer=infer, asr=OfflineASR(), tts=TTSOutput())
        ui_state = UIState(status="active", mode=main_runtime.mode, target=main_runtime.learn_target)
        web = build_web_server(cfg.web.host, cfg.web.port, main_runtime, ui_state)
        web_thread = Thread(target=web.serve_forever, daemon=True)
        web_thread.start()

        if args.speech:
            result = main_runtime.run_speech_to_sign(args.speech)
            ui_state.transcript = str(result["transcript"])
            ui_state.letters = list(result["letters"])

        print(f"Web UI: {cfg.web.kiosk_url}")
        if cfg.web.auto_open_browser:
            _open_local_ui(cfg.web.kiosk_url, kiosk_mode=cfg.web.kiosk_mode)
        try:
            main_runtime.run_camera_loop()
        finally:
            web.shutdown()
        return

    if args.run_daemon:
        daemon = StandbyDaemon(config=cfg)
        if args.mock_serial:
            events = [item.strip() for item in args.mock_serial.split(",") if item.strip()]
            for line in events:
                event = parse_serial_line(line, cfg.serial)
                if event:
                    print(f"event={line} action={daemon.apply_serial_event(event)} state={daemon.state_machine.state.name}")
                print(f"tick={daemon.tick()} state={daemon.state_machine.state.name}")
            return
        daemon.run_forever()
        return

    print("GestureBridge ready. Use --run-daemon / --run-main / --benchmark-asl29.")


if __name__ == "__main__":
    main()
