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
from gesturebridge.system.mic_default import prefer_c270_default_mic


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
    for browser_cmd in ("chromium-browser", "chromium"):
        browser_bin = shutil.which(browser_cmd)
        if not browser_bin:
            continue
        if kiosk_mode:
            subprocess.Popen([browser_bin, "--kiosk", "--noerrdialogs", "--disable-infobars", url])
            print(f"Opened kiosk browser: {browser_cmd}")
            return
        subprocess.Popen([browser_bin, "--new-window", url])
        print(f"Opened windowed browser: {browser_cmd}")
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
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="override cfg.asl29.runtime.camera_index (e.g. 0 = built-in webcam, 1 = external; on Pi C270 lands at /dev/video0 → index 0)",
    )
    args = parser.parse_args()
    cfg = SystemConfig()
    if args.camera_index is not None:
        cfg.asl29.runtime.camera_index = args.camera_index
        print(f"[app] camera_index override: {args.camera_index}")
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
        # Optional landmark MLP — only attached if its TFLite file exists. The
        # ensemble rule lives in MainRuntime._maybe_ensemble.
        landmark_classifier = None
        landmark_mlp_dir = Path("artifacts/asl29/landmark_mlp")
        for _lm_name in ("landmark_mlp.npz", "landmark_mlp.tflite"):
            _lm_path = landmark_mlp_dir / _lm_name
            if _lm_path.exists():
                from gesturebridge.pipelines.landmark_classifier import LandmarkClassifier
                landmark_classifier = LandmarkClassifier(
                    model_path=_lm_path,
                    labels_path=Path(cfg.asl29.data.labels_path),
                )
                print(f"[app] landmark MLP attached: {_lm_path}")
                break
        # WLASL-100 word classifier auto-attach. Precedence (highest first):
        #   1. A100 5-way ensemble: 3x BigConv1D (s42/s43/s1337) + Conv1D + GRU
        #      with 0.7 weight on A100 swarm and 0.3 on existing pair.
        #      Honest test: top-1 0.674 / top-5 0.921 (vs deployed 0.577/0.870).
        #   2. A100 BigConv1D swarm only (3x mean): 0.657 / 0.908.
        #   3. Existing Conv1D + GRU ensemble (deployed prior): 0.577 / 0.870.
        #   4. Conv1D alone, then None.
        #
        # Auto-attach is silent / falls through if files are missing (Pi may
        # not have the bigconv1d_*.npz triple yet on first deploy).
        word_classifier = None
        word_labels = Path("artifacts/wlasl100/labels.txt")
        word_npz = Path("artifacts/wlasl100/conv1d_small.npz")
        gru_npz = Path("artifacts/wlasl100/gru_small.npz")
        bigconv_paths = [
            Path("artifacts/wlasl100/bigconv1d_s42.npz"),
            Path("artifacts/wlasl100/bigconv1d_s43.npz"),
            Path("artifacts/wlasl100/bigconv1d_s1337.npz"),
        ]
        if word_labels.exists():
            try:
                from gesturebridge.pipelines.word_classifier import WordClassifier

                bigconv_classifiers = []
                for p in bigconv_paths:
                    if p.exists():
                        from gesturebridge.pipelines.word_bigconv1d import BigConv1DClassifier

                        bigconv_classifiers.append(BigConv1DClassifier(model_path=p, labels_path=word_labels))

                conv = WordClassifier(model_path=word_npz, labels_path=word_labels) if word_npz.exists() else None
                gru = None
                if gru_npz.exists():
                    from gesturebridge.pipelines.word_ensemble import GRUClassifier

                    gru = GRUClassifier(model_path=gru_npz, labels_path=word_labels)

                if bigconv_classifiers and conv is not None and gru is not None:
                    from gesturebridge.pipelines.word_ensemble import MultiEnsembleWordClassifier

                    # 0.7 / 0.3 split: each BigConv1D gets 0.7/3, Conv1D and GRU each get 0.15.
                    bc_w = 0.7 / max(1, len(bigconv_classifiers))
                    members = [(bc, bc_w) for bc in bigconv_classifiers]
                    members += [(conv, 0.15), (gru, 0.15)]
                    word_classifier = MultiEnsembleWordClassifier(members=members)
                    print(
                        f"[app] WLASL-100 5-way ensemble attached "
                        f"({len(bigconv_classifiers)}x BigConv1D + Conv1D + GRU, "
                        f"{len(conv.labels)} classes; expected test top-1 ~0.67)"
                    )
                elif bigconv_classifiers:
                    from gesturebridge.pipelines.word_ensemble import MultiEnsembleWordClassifier

                    members = [(bc, 1.0 / len(bigconv_classifiers)) for bc in bigconv_classifiers]
                    word_classifier = MultiEnsembleWordClassifier(members=members)
                    print(f"[app] WLASL-100 BigConv1D swarm attached ({len(bigconv_classifiers)}x)")
                elif conv is not None and gru is not None:
                    from gesturebridge.pipelines.word_ensemble import EnsembleWordClassifier

                    word_classifier = EnsembleWordClassifier(conv=conv, gru=gru)
                    print(
                        f"[app] WLASL-100 Conv1D+GRU ensemble attached ({len(conv.labels)} classes)"
                    )
                elif conv is not None:
                    word_classifier = conv
                    print(f"[app] WLASL-100 Conv1D alone attached ({len(conv.labels)} classes)")
            except Exception as exc:
                import traceback
                print(f"[app] word_classifier load failed: {exc}", flush=True)
                traceback.print_exc()
        main_runtime = MainRuntime(
            config=cfg,
            infer=infer,
            asr=OfflineASR(),
            tts=TTSOutput(),
            landmark_classifier=landmark_classifier,
            word_classifier=word_classifier,
        )
        # Load the global confidence threshold so low-confidence captures
        # are flagged "ambiguous" instead of being shown as a top-1 label.
        calib_path = Path("artifacts/wlasl100/calibration.npz")
        if calib_path.exists():
            try:
                cal = np.load(calib_path, allow_pickle=True)
                main_runtime._word_threshold = float(cal["global_threshold"][0])
                main_runtime._word_calibration_meta = {
                    "test_top1": float(cal["test_top1"][0]),
                    "test_top5": float(cal["test_top5"][0]),
                    "test_gated_coverage": float(cal["test_gated_coverage_global"][0]),
                    "test_gated_precision": float(cal["test_gated_precision_global"][0]),
                }
                print(
                    f"[app] confidence gate loaded: threshold={main_runtime._word_threshold:.3f} "
                    f"(test gated: cov={main_runtime._word_calibration_meta['test_gated_coverage']:.2f}, "
                    f"prec={main_runtime._word_calibration_meta['test_gated_precision']:.2f})"
                )
            except Exception as exc:
                print(f"[app] calibration load failed (non-fatal): {exc}", flush=True)
        ui_state = UIState(status="active", mode=main_runtime.mode, target=main_runtime.learn_target)
        web = build_web_server(cfg.web.host, cfg.web.port, main_runtime, ui_state)
        web_thread = Thread(target=web.serve_forever, daemon=True)
        web_thread.start()

        if args.speech:
            result = main_runtime.run_speech_to_sign(args.speech)
            ui_state.transcript = str(result["transcript"])
            ui_state.letters = list(result["letters"])

        print(f"Web UI: {cfg.web.kiosk_url}")
        prefer_c270_default_mic()
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
