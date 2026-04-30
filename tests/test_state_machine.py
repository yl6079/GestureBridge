from gesturebridge.state_machine import ModeState, SystemState, SystemStateMachine


def test_state_machine_wakeup_and_sleep_cycle() -> None:
    sm = SystemStateMachine()
    assert sm.state == SystemState.IDLE_LOW_POWER
    assert sm.on_activity_detected(score=0.9, threshold=0.5) is True
    assert sm.state == SystemState.WAKE_REQUESTED
    sm.on_wake_ack()
    assert sm.state == SystemState.ACTIVE
    sm.on_mode_selected(ModeState.TRANSLATE_SIGN_TO_SPEECH)
    assert sm.mode == ModeState.TRANSLATE_SIGN_TO_SPEECH
    sm.on_inactivity_timeout()
    assert sm.state == SystemState.COOLDOWN
    sm.on_sleep_complete()
    assert sm.state == SystemState.IDLE_LOW_POWER
    assert sm.mode == ModeState.MODE_SELECT
