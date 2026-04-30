from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gesturebridge.config import SystemConfig
from gesturebridge.devices.rpi import RPiRuntime
from gesturebridge.devices.xiao import XIAODetector
from gesturebridge.modes.learn import LearnMode
from gesturebridge.modes.translate import TranslateMode
from gesturebridge.state_machine import ModeState, SystemStateMachine


@dataclass(slots=True)
class GestureBridgeController:
    config: SystemConfig
    xiao: XIAODetector
    rpi: RPiRuntime
    state_machine: SystemStateMachine
    translate_mode: TranslateMode
    learn_mode: LearnMode

    def wake_if_needed(self, activity_level: float) -> str:
        event = self.xiao.detect_activity(activity_level)
        if event is None:
            return "no_activity"
        should_wake = self.state_machine.on_activity_detected(
            score=event.activity_level,
            threshold=self.config.thresholds.activity_trigger,
        )
        if not should_wake:
            return "already_active"
        ack = self.rpi.handle_wake(event)
        if ack == "wake_ack":
            self.state_machine.on_wake_ack()
        return ack

    def run_translate_sign_to_speech(self, frame: np.ndarray) -> dict[str, str | float]:
        self.state_machine.on_mode_selected(ModeState.TRANSLATE_SIGN_TO_SPEECH)
        self.rpi.touch()
        return self.translate_mode.sign_to_speech(frame)

    def run_translate_speech_to_sign(self, speech_input: str) -> dict[str, str]:
        self.state_machine.on_mode_selected(ModeState.TRANSLATE_SPEECH_TO_SIGN)
        self.rpi.touch()
        return self.translate_mode.speech_to_sign(speech_input)

    def run_learn_teaching(self, frame: np.ndarray, target_sign_id: int) -> dict[str, str | float | bool]:
        self.state_machine.on_mode_selected(ModeState.LEARN_TEACHING)
        self.rpi.touch()
        return self.learn_mode.teaching_stage(frame, target_sign_id)

    def run_learn_practice(self, frame: np.ndarray, target_sign_id: int) -> dict[str, str | float | bool]:
        self.state_machine.on_mode_selected(ModeState.LEARN_PRACTICE)
        self.rpi.touch()
        return self.learn_mode.practice_stage(frame, target_sign_id)

    def housekeeping(self) -> str:
        slept = self.rpi.maybe_sleep(self.config.thresholds.inactivity_seconds)
        if slept:
            self.state_machine.on_inactivity_timeout()
            self.state_machine.on_sleep_complete()
            return "sleep_complete"
        return "awake"
