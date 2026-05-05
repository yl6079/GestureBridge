# Yizheng WeChat 同步记录

每次迭代结束后，把要请 Yizheng 帮忙或同步的内容写在这里。微信风格，简短，
中文。Shufeng 直接复制粘贴发给 Yizheng。

---

## 第 1 次同步 — 2026-05-05

哥，进展同步：

教授要求加上 dynamic gesture（动作单词识别），不只是字母。现在的 speech-to-sign 模式
你应该见过了，比如说 "electrical" 它会蹦 10 张字母图片，太傻了。我已经先把这个口子
堵了 —— 第一版只支持 5 个常用词（hello / thanks / yes / no / help），UI 改成播 1-2 秒
的视频片段，词不在词库的词还是按字母拼。视频是从 aslbricks.org 和 signasl.org 抓的，
没传到 git 里（版权不明），脚本在 `scripts/fetch_word_clips.sh`，你那边要复现的话
跑一下就行。

接下来三天我会按 iteration 来，每改一版你都能直接测。下一步是抓 WLASL-100 的数据
集，训一个 Conv1D 跑动作识别，再把它接进新的 Word 模式。字母模式保留，作为兜底。

需要你帮忙（不急但都要）：

1. 树莓派的 IP 或者主机名给我一下（局域网内的就行），这样我可以直接 ssh 上去 deploy。
   账号密码我已经有了（elen6908 / group16）。
2. C270 摄像头有没有插好，跑一下 `v4l2-ctl --list-devices` 看看显示的设备名，
   截个图发我。
3. ESP32 现在挂在 Pi 哪个 USB 上？`ls /dev/ttyUSB* /dev/ttyACM*` 看一下，确认能
   读到 `Hand:` / `Empty:` 这种字符串就行。
4. Pi 上的 venv 在哪？`which python` 给我贴一下路径。

不急的：
- 你之前改的 web UI timeout backoff 我没动，改完 word mode 之后我会跑个回归
  确认还稳定。

迭代日志在 `notes/iteration_log.md`，要看技术细节去那个文件。

---

## 第 2 次同步 — 2026-05-05 下午

哥，又一波进展同步。

我用 Tailscale 连了下树莓派（IP 100.127.215.9，从 `notes/pi_access.md` 找到的）：

- ✅ Pi 在线，C270 摄像头识别成 `/dev/video0`，没问题。
- ✅ USB 音箱（JieLi 那个）也识别上了。
- ⚠️ **ESP32 现在树莓派看不到**，`/dev/ttyUSB*` 和 `/dev/ttyACM*` 都没有，
  `dmesg` 也没有相关日志。你那条 "esp32 插中间靠里面" 我已经记下来
  了，但我猜要么没插紧、要么固件没烧成功。**你下次到 Pi 旁边的时候帮
  忙看一下**。Word 模式不依赖 ESP32（wake gating 那一层我会让它在没
  serial 信号时 fallback 成 always-on），所以这块不是 P0。

我做完的（你可以拉 shufeng 分支自己看）：

1. **Speech-to-sign 视频化**（IT-1）：5 个常用词（hello/thanks/yes/no/help）
   现在播视频。其他词降级回字母拼写。`scripts/fetch_word_clips.sh` 可以
   重新下载（视频文件没传 git，版权不明）。
2. **WLASL-100 数据集抓取**（IT-2）：100 个常用词，676 个片段，每个词
   至少 3 个片段。`scripts/prepare_wlasl100.py`。所有数据 gitignored。
3. **Conv1D 词识别模型训练**（IT-3）：(30 帧 × 63 维 landmark) → 100 类。
   验证集 top-1 21%，top-5 53%。说实话不算高，因为每类只有 5 个训练样
   本。但是某些信号清晰的词比如 "yes" 在自分布数据上 88% 置信。我会
   找时间再训一版（更多 augmentation + YouTube 数据补强）。
4. **--camera-index N CLI 标志**（IT-5）：可以切换摄像头。Mac 上用
   built-in 测试时 `--camera-index 0`，Pi 上 C270 也是 0。
5. **`predict_word_clip.py`**：你给它一个 MP4，它告诉你 top-5 是哪些
   词。我用这个做了几个 spot check。
6. 把 `scripts/deploy_to_pi.sh` 默认路径改成了
   `/home/elen6908/Documents/GestureBridge`（没有 `/test` 后缀了，
   你之前移过的），如果你之前用别的环境变量覆盖过就没影响。

下一步我打算做的（不阻塞你）：

- IT-4：在 Web UI 加一个 "Word" tab，按 "Capture" 按钮拍 1 秒，跑模型，
  显示 top-5。Mac 上先调通。
- 用 yt-dlp 把 WLASL 的 YouTube 那部分也抓下来（直接源只有 33% 命中），
  数据量大约能翻倍。然后再 train 一次，应该能把 val top-1 推到 30%+。

不急但需要你的事（你方便了告诉我就行，不用回信）：

1. **ESP32 现在状态** —— 到 Pi 旁边时戳一下 USB，看 `ls /dev/ttyUSB*
   /dev/ttyACM*` 有没有东西。如果固件需要重烧，告诉我，我们 IT-6/7
   再处理。
2. Pi 上的 venv 路径 (`which python` 在你常用的环境里贴一下) —— 这样我
   deploy 之后跑 `python -m gesturebridge.app --run-main` 不会用错环境。

---


