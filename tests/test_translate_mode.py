import numpy as np

from gesturebridge.bootstrap import build_controller


def test_translate_speech_to_sign_known_and_unknown() -> None:
    controller = build_controller()
    controller.wake_if_needed(0.9)
    known = controller.run_translate_speech_to_sign("hello")
    assert known["status"] == "OK"
    assert known["meaning"] == "hello"

    unknown = controller.run_translate_speech_to_sign("this is not in vocabulary")
    assert unknown["status"] == "NO_MATCH"


def test_translate_sign_to_speech_returns_result_dict() -> None:
    controller = build_controller()
    controller.wake_if_needed(0.9)
    frame = np.random.default_rng(1).normal(size=(21, 3)).astype(np.float32)
    result = controller.run_translate_sign_to_speech(frame)
    assert "status" in result
    assert "confidence" in result
