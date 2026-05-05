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

