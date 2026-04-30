# Yizheng WeChat update — 2026-04-30 evening

## What he said (translated)

### 16:58 — three messages

**1. About the listener program & model performance (current Pi state):**
- The listener program (web app daemon) is **not yet set up to auto-start on
  boot** — has to be launched manually after every reboot.
- He attributes the poor real-world accuracy to **"dataset and test environment
  mismatch"** — on the held-out test set the accuracy "is pretty high"
  (consistent with our finding that his eval ran on the leaky split → fake 100%).
- The `prediction_confidence` threshold was originally **0.65**. With the
  deployed model, almost nothing crossed it.
- Easier gestures (W, Y) sit in **top-1** stably but with confidence only
  **0.08–0.09**.
- He temporarily **lowered the threshold to 0.08** "to get the flow working,"
  acknowledging it needs to be raised again later.

**2. Proposed division of labor:**
> "How about you try mediapipe + landmark MLP on your side, while I retrain
> the model on my side?"

**3. On INT8 vs FP32:**
> "I don't think we even need to chase INT8 — FP32 on the Raspberry Pi is
> totally fine, no need to quantize further."

### 17:03 — USB port standardization request

He wants the three USB devices in fixed slots so cabling/`/dev/*` paths are
predictable:
- **Speaker** → bottom-inner USB port
- **Camera (C270)** → bottom-outer USB port
- **ESP32** → middle-inner USB port

(User replied at 17:14: "okay, I'll look into it.")

### 17:14 — sent assignment deliverables doc

Reminder of what's due **Monday 2026-05-04** (4 days from now):

- **70% — Final Deliverables:**
  1. Video recording link
  2. GitHub repo link (with README)
  3. Zip: slides + source + any additional materials
  4. ~4-page ACM double-column report (figures/tables OK)
- **20% — Demo + Q&A:**
  1. 3-minute demo video (≤3 min, **mandatory**)
  2. Q&A (5%)
  3. Peer voting (5% extra credit)
- Demo video must: demonstrate the system, highlight key challenges,
  explain important design decisions.
- At least one team member at the live Q&A.

### Other artifacts shared

- `esp32_camera.ino` (11.8 KB) — the **currently-flashed** ESP32 firmware.
  Needs to be checked into the repo (probably `firmware/`).
- Hardware photos showing the Pi + display + USB devices arrangement.

## Where his picture matches / doesn't match ours

| Topic | Yizheng's position | Our finding | Verdict |
|---|---|---|---|
| Why model fails in real life | "dataset / test env mismatch" | Same root cause: split leakage + Kaggle distribution ≠ C270, plus chirality flip bug in augmentation | **Aligned** (we have more granular root cause) |
| What to do next | He retrains; you do mediapipe + landmark MLP | We already did **both**: new model val=88.6%, landmark MLP added, ensemble 82.9% on honest test | **He's about to duplicate work — flag immediately** |
| INT8 | "FP32 on Pi is fine, no need to quantize" | Our INT8 = 21.8% (broken calibration). FP32 = 80.2%. We agree skip INT8 for the demo. | **Aligned** |
| Confidence threshold | Lowered to 0.08 because old model's real-world confidence was 0.08–0.09 | We set 0.4 in config (and noted in code why). With the new hand-cropped model, real-world confidence should be much higher — needs Pi verification before locking | **Need to verify after deploy** |
| USB standardization | Requested specific port layout | No code change needed, but check `/dev/ttyACM0`, camera index, audio device aren't hard-coded to wrong assumptions | **Aligned, low risk** |
| Auto-start on boot | Currently manual | Not in our scope yet; could add a systemd unit | **Open** |
| ESP32 firmware | Only on the device + as a `.ino` he just sent | Not in repo | **Open** — ask him to push or we add the file |

## Bottom line

We are **ahead** of where he thinks we are. He is about to retrain a model
that we have already retrained (better — honest 80.2% FP32 / 82.9% ensemble
vs. his fake 100%). We need to tell him **before** he wastes 4090 budget /
his time re-doing the work.

The biggest immediate risks:
1. He retrains in parallel → divergent / duplicated work.
2. His current Pi has confidence threshold 0.08 — once we deploy the new
   model, that needs to come back up (our config defaults to 0.4) or every
   noise frame will fire.
3. Demo is 4 days away. Need to deploy + demo-record + write 4-page report.
