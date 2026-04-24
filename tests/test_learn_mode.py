import numpy as np

from gesturebridge.bootstrap import build_controller


def test_learn_mode_updates_stats() -> None:
    controller = build_controller()
    controller.wake_if_needed(0.9)
    frame = np.random.default_rng(2).normal(size=(21, 3)).astype(np.float32)

    before = controller.learn_mode.stats.attempts
    controller.run_learn_teaching(frame, target_sign_id=0)
    controller.run_learn_practice(frame, target_sign_id=1)
    after = controller.learn_mode.stats.attempts
    assert after == before + 2
