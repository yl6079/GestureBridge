# Test Matrix

## Functional

- Translate sign-to-speech pipeline returns spoken output above confidence threshold.
- Translate speech-to-sign resolves known keywords and handles unknown commands.
- Learn teaching stage returns immediate correctness feedback.
- Learn practice stage evaluates sign against meaning-only prompt.
- State machine transitions: idle -> wake_requested -> active -> cooldown -> idle.

## Performance

- Average per-sample inference latency from quantized classifier.
- End-to-end response path timing in demo script.
- Memory and CPU checks during run on Raspberry Pi target.

## Reliability

- Repeated wake cycles do not deadlock.
- Inactivity timeout triggers sleep recovery.
- No unhandled exception during long-run loop.

## Power

- Compare always-on baseline vs wake-on-demand simulation.
- Track wake count and idle/active transitions.
