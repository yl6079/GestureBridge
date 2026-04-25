# XIAO ESP32S3 — Edge Impulse Firmware

This folder holds the deployment artifact for the **XIAO ESP32S3** front-end device.

---

## Edge Impulse project

| Field | Value |
|-------|-------|
| Project | ELEN6908Project |
| Project ID | 959410 |
| Studio URL | https://studio.edgeimpulse.com/studio/959410 |
| Model | Binary hand detection (Hand vs. Empty) |
| Training data | 97,628 images (89,628 Hand + 8,000 Empty) |
| Accuracy | Trained and ready to deploy |

---

## How to download the firmware (Shufeng / Yizheng)

1. Open https://studio.edgeimpulse.com/studio/959410
2. Click **Deployment** in the left sidebar.
3. Under **Build firmware**, select target:
   - **Espressif ESP-EYE** (closest to XIAO ESP32S3)
   - *or* **Arduino library** if you want to integrate into a sketch.
4. Click **Build**.
5. Download the `.zip` file when ready.
6. Place the downloaded `.zip` here (this folder).

---

## How to flash (Yizheng — physically in his office)

### Option A — Arduino IDE (recommended)

```bash
# 1. Install Arduino IDE 2.x from https://www.arduino.cc/en/software
# 2. Add Espressif board support:
#    Arduino IDE → File → Preferences → Additional boards manager URLs:
#    https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
# 3. Install "esp32 by Espressif" from Boards Manager
# 4. Open the firmware .ino sketch from the downloaded zip
# 5. Select board: Tools → Board → ESP32 Arduino → XIAO_ESP32S3
# 6. Select port: the USB-C port the XIAO is connected to
# 7. Click Upload (→)
```

### Option B — esptool.py (command line)

```bash
pip install esptool

# Put XIAO in boot mode:
# Hold BOOT button, press RESET, release RESET, release BOOT.

# Flash the firmware binary (.bin from the zip):
esptool.py --chip esp32s3 --port /dev/ttyUSB0 write_flash 0x0 firmware.bin

# Replace /dev/ttyUSB0 with the actual port (COMx on Windows, /dev/cu.* on Mac).
```

---

## After flashing

Once flashed, the XIAO will:
1. Continuously capture frames from its **OV2640** camera.
2. Run the Edge Impulse hand detection model.
3. On a positive detection, send a **wake signal** to the Raspberry Pi via **USB serial**.

The Raspberry Pi listens for this signal in `src/gesturebridge/devices/xiao.py`
(currently a stub — pyserial integration is the **next iteration**).

---

## Notes

- The Edge Impulse firmware already contains the trained model weights — no
  retraining or Edge Impulse account needed to flash.
- The XIAO is powered via USB-C from the Pi (or any USB port).
- Serial baud rate: **115200** (default for Edge Impulse firmware).
