from __future__ import annotations

import argparse

import numpy as np

from gesturebridge.bootstrap import build_controller


def run_demo() -> None:
    controller = build_controller()
    print("Wake:", controller.wake_if_needed(activity_level=0.9))

    sample_frame = np.random.default_rng(42).normal(size=(21, 3)).astype(np.float32)
    print("Translate sign->speech:", controller.run_translate_sign_to_speech(sample_frame))
    print("Translate speech->sign:", controller.run_translate_speech_to_sign("hello"))

    print("Learn teaching:", controller.run_learn_teaching(sample_frame, target_sign_id=0))
    print("Learn practice:", controller.run_learn_practice(sample_frame, target_sign_id=1))
    print("Housekeeping:", controller.housekeeping())


def main() -> None:
    parser = argparse.ArgumentParser(description="GestureBridge runtime demo")
    parser.add_argument("--demo", action="store_true", help="run quick pipeline demo")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    print("GestureBridge ready. Run with --demo to execute sample flows.")


if __name__ == "__main__":
    main()
