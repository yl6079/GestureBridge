#!/usr/bin/env bash
# 外接 USB 摄像头 + 扬声器快速自检（Linux / 树莓派适用）
# Quick check for external USB camera and speakers. Requires: ffmpeg, aplay.
#
# 用法 Usage:
#   ./scripts/test_camera_speaker.sh              # 列出设备并运行默认测试
#   ./scripts/test_camera_speaker.sh --list       # 仅列出设备
#   PLAYBACK=plughw:3,0 VIDEO=/dev/video0 ./scripts/test_camera_speaker.sh
#
# 环境变量 Env:
#   PLAYBACK  aplay -D 的设备。未设置时自动选「播放设备列表里带 USB 的声卡」
#             （树莓派上全局 default 常指向 HDMI，无显示器时会 aplay 报错 524）
#   VIDEO     V4L2 设备路径（默认自动选：v4l2 列表里第一个非 pisp/rpi-hevc 的 capture 节点）
#   WIDTH HEIGHT  采集分辨率（默认 640x480）

set -euo pipefail

# 空字符串表示「自动」；若需强制用系统 default： PLAYBACK=default
PLAYBACK="${PLAYBACK-}"
VIDEO="${VIDEO-}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"

LIST_ONLY=0
SKIP_SOUND=0
SKIP_CAMERA=0
PREVIEW=0

usage() {
  sed -n '1,20p' "$0" | grep -E '^# ' | sed 's/^# //'
  echo
  echo "选项 Options:"
  echo "  --list          只列出音视频设备"
  echo "  --no-sound      跳过扬声器"
  echo "  --no-camera     跳过摄像头"
  echo "  --preview       用 ffplay 打开预览（需要图形界面）"
  echo "  -h, --help      帮助"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) LIST_ONLY=1; shift ;;
    --no-sound) SKIP_SOUND=1; shift ;;
    --no-camera) SKIP_CAMERA=1; shift ;;
    --preview) PREVIEW=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
  esac
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "缺少命令: $1（请安装对应软件包）" >&2
    exit 1
  }
}

# 在 PLAYBACK 段落里找第一个名字里含 USB 的播放卡（外接 USB 音箱常见）
guess_usb_playback() {
  local card
  card=$(
    aplay -l 2>/dev/null |
      sed -n '/PLAYBACK Hardware/,/CAPTURE Hardware/p' |
      grep -E '^card[[:space:]]+[0-9]+:' |
      grep -i usb |
      head -1 |
      awk '{ gsub(/:/, "", $2); print $2; exit }'
  )
  [[ -n "${card:-}" ]] && echo "plughw:${card},0"
}

resolve_playback() {
  if [[ -n "${PLAYBACK}" ]]; then
    echo "$PLAYBACK"
    return
  fi
  local usb
  usb=$(guess_usb_playback || true)
  if [[ -n "${usb}" ]]; then
    echo "$usb"
    return
  fi
  echo "default"
}

# 优先用 v4l2-ctl 里标注为摄像头的那一路 /dev/videoN（避免误选 pisp 管道节点）
guess_v4l2_capture_device() {
  if ! command -v v4l2-ctl >/dev/null 2>&1; then
    echo "/dev/video0"
    return
  fi
  local dev
  dev=$(
    v4l2-ctl --list-devices 2>/dev/null |
      awk '
        /^[^[:space:]]/ { if ($0 ~ /pispbe|rpi-hevc-dec/) skip=1; else skip=0 }
        /^[[:space:]]+\/dev\/video/ && skip==0 {
          if (match($0,/\/dev\/video[0-9]+/)) { print substr($0,RSTART,RLENGTH); exit }
        }
      '
  )
  [[ -n "${dev}" && -e "${dev}" ]] && echo "$dev" && return
  echo "/dev/video0"
}

resolve_video() {
  if [[ -n "${VIDEO}" ]]; then
    echo "$VIDEO"
    return
  fi
  guess_v4l2_capture_device
}

list_devices() {
  echo "======== V4L2 视频设备 /dev/video* ========"
  if ls /dev/video* 2>/dev/null; then
    ls -l /dev/video* 2>/dev/null || true
  else
    echo "(未找到 /dev/video*)"
  fi
  if command -v v4l2-ctl >/dev/null 2>&1; then
    echo
    echo "-------- v4l2-ctl --list-devices --------"
    v4l2-ctl --list-devices 2>/dev/null || true
  fi

  echo
  echo "======== ALSA 播放设备 aplay -l ========"
  aplay -l 2>/dev/null || echo "(aplay 不可用)"

  echo
  echo "======== ALSA 录音设备 arecord -l ========"
  arecord -l 2>/dev/null || echo "(arecord 不可用)"

  echo
  echo "-------- aplay 可用 PCM 名 aplay -L（节选）--------"
  aplay -L 2>/dev/null | head -n 40 || true
}

test_speaker() {
  need_cmd ffmpeg
  need_cmd aplay
  echo
  echo ">>> 扬声器测试：约 0.35s 880Hz 正弦波（设备: $PLAYBACK）"
  if ! ffmpeg -hide_banner -loglevel error -nostdin \
    -f lavfi -i "sine=frequency=880:duration=0.35" \
    -ac 1 -ar 48000 -f wav - 2>/dev/null \
    | aplay -q -D "$PLAYBACK" -; then
    echo "    aplay 失败。可尝试: export PLAYBACK=plughw:CARD=Device,DEV=0" >&2
    return 1
  fi
  echo "    若未听到声音，请换 PLAYBACK，例如: export PLAYBACK=plughw:3,0"
}

test_camera_preview() {
  need_cmd ffplay
  echo
  echo ">>> 摄像头预览（Ctrl+C 退出）设备: $VIDEO"
  ffplay -hide_banner -loglevel error -f v4l2 -input_format mjpeg \
    -video_size "${WIDTH}x${HEIGHT}" -i "$VIDEO"
}

test_camera_still() {
  need_cmd ffmpeg
  local out="/tmp/gesturebridge_cam_test.jpg"
  echo
  echo ">>> 摄像头单帧 -> $out （$VIDEO ${WIDTH}x${HEIGHT} MJPEG）"
  ffmpeg -hide_banner -loglevel error -nostdin -y \
    -f v4l2 -input_format mjpeg -video_size "${WIDTH}x${HEIGHT}" \
    -i "$VIDEO" -frames:v 1 -q:v 2 "$out"
  echo "    请打开图片确认画面是否正常。"
}

main() {
  need_cmd aplay
  PLAYBACK=$(resolve_playback)
  VIDEO=$(resolve_video)

  list_devices
  echo
  echo "-------- 本次测试选用 Resolved --------"
  echo "PLAYBACK=$PLAYBACK  (设置 PLAYBACK= 可覆盖；留空则自动 USB 优先)"
  echo "VIDEO=$VIDEO  (设置 VIDEO= 可覆盖；留空则尽量从 v4l2 推断摄像头)"

  [[ "$LIST_ONLY" -eq 1 ]] && exit 0

  if [[ "$SKIP_SOUND" -eq 0 ]]; then
    test_speaker || exit 1
  fi

  if [[ "$SKIP_CAMERA" -eq 0 ]]; then
    if [[ ! -e "$VIDEO" ]]; then
      echo "ERROR: 摄像头设备不存在: $VIDEO" >&2
      echo "请设置 VIDEO，例如: export VIDEO=/dev/video1" >&2
      exit 1
    fi
    if [[ "$PREVIEW" -eq 1 ]]; then
      test_camera_preview
    else
      test_camera_still
    fi
  fi

  echo
  echo "完成。也可用项目自带 Python 自检:"
  echo "  python scripts/hardware_smoke_test.py --alsa-playback \"\$PLAYBACK\" --camera-device \"\$VIDEO\""
}

main
