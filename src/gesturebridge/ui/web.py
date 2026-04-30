from __future__ import annotations

import json
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

from gesturebridge.system.main_runtime import MainRuntime


@dataclass(slots=True)
class UIState:
    mode: str = "read"
    status: str = "standby"
    prediction: str = "-"
    confidence: float = 0.0
    transcript: str = ""
    letters: list[str] = field(default_factory=list)
    target: str = "A"
    passed: bool = False
    tts: str = ""
    sign_assets: list[str] = field(default_factory=list)


def _index_html() -> str:
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>GestureBridge</title>
<style>
*{box-sizing:border-box}
body{
  font-family:Inter,Segoe UI,Arial,sans-serif;
  margin:0;
  background:linear-gradient(120deg,#0b1220,#111827 45%,#0f172a);
  color:#e5e7eb;
  min-height:100vh;
}
.wrap{max-width:980px;margin:24px auto;padding:0 18px}
.topbar{
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:14px;
}
.title{font-size:28px;font-weight:700;letter-spacing:.4px}
.badge{
  font-size:13px;padding:8px 12px;border-radius:999px;
  background:#1f2937;border:1px solid #374151;color:#cbd5e1;
}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 8px}
button{
  padding:11px 14px;border-radius:10px;border:1px solid #334155;
  background:#111827;color:#e5e7eb;cursor:pointer;
  transition:all .15s ease;
}
button:hover{background:#1f2937;border-color:#475569}
button.active{background:#2563eb;border-color:#2563eb;color:#fff}
button:disabled{opacity:.6;cursor:wait}
.grid{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:12px;
  margin-top:14px;
}
.card{
  background:rgba(17,24,39,.85);
  border:1px solid #334155;
  border-radius:14px;
  padding:14px;
  min-height:82px;
}
.hidden{display:none}
.preview{
  width:100%;
  border-radius:14px;
  border:1px solid #334155;
  background:#000;
  min-height:260px;
  object-fit:cover;
}
.sign-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
  gap:10px;
  margin-top:12px;
}
.sign-item{
  background:rgba(17,24,39,.85);
  border:1px solid #334155;
  border-radius:12px;
  padding:10px;
  text-align:center;
}
.sign-item img{
  width:100%;
  max-height:110px;
  object-fit:contain;
  border-radius:8px;
  background:#0b1220;
}
.sign-item .lbl{
  margin-top:6px;
  font-size:13px;
  color:#cbd5e1;
}
.k{font-size:12px;color:#93c5fd;margin-bottom:6px;text-transform:uppercase;letter-spacing:.08em}
.v{font-size:19px;font-weight:600;word-break:break-word}
.hint{margin-top:12px;color:#93c5fd;font-size:13px;min-height:20px}
</style></head><body>
<div class="wrap">
  <div class="topbar">
    <div class="title">GestureBridge</div>
    <div class="badge" id="currentMode">Mode: read</div>
  </div>
  <div class="toolbar">
    <button id="btn-read" onclick="setMode('read')">Read Mode</button>
    <button id="btn-speech_to_sign" onclick="setMode('speech_to_sign')">Speech to Sign</button>
    <button id="btn-learn" onclick="setMode('learn')">Learning Practice</button>
  </div>
  <img id="preview" class="preview" src="/video.jpg" alt="Camera preview" />
  <div class="card hidden" id="card-learn-target-display">
    <div class="k">Learning Target Sign</div>
    <div style="display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap">
      <button id="btn-learn-prev" onclick="shiftLearnTarget(-1)">Prev</button>
      <img id="learnTargetImage" src="" alt="Target sign" style="width:180px;max-width:42vw;max-height:180px;object-fit:contain;border-radius:10px;background:#0b1220;border:1px solid #334155;" />
      <button id="btn-learn-next" onclick="shiftLearnTarget(1)">Next</button>
    </div>
  </div>
  <div class="grid">
    <div class="card" id="card-status"><div class="k">Status</div><div class="v" id="status">-</div></div>
    <div class="card" id="card-prediction"><div class="k">Prediction</div><div class="v" id="prediction">-</div></div>
    <div class="card" id="card-confidence"><div class="k">Confidence</div><div class="v" id="confidence">0.00</div></div>
    <div class="card" id="card-tts"><div class="k">TTS</div><div class="v" id="tts">-</div></div>
    <div class="card" id="card-transcript"><div class="k">Transcript</div><div class="v" id="transcript">-</div></div>
    <div class="card" id="card-letters"><div class="k">Letters</div><div class="v" id="letters">-</div></div>
    <div class="card" id="card-target"><div class="k">Target</div><div class="v" id="target">-</div></div>
    <div class="card" id="card-passed"><div class="k">Passed</div><div class="v" id="passed">-</div></div>
  </div>
  <div class="card hidden" id="card-sign-gallery">
    <div class="k">Sign Images</div>
    <div id="signGallery" class="sign-grid"></div>
  </div>
  <div class="hint" id="hint">Ready.</div>
</div>
<script>
let refreshing = false;
let speechRecognizer = null;
let speechRunning = false;
let speechSupported = false;
function setHint(msg){ document.getElementById('hint').textContent = msg; }
function updateLearnTargetImage(target){
  const img = document.getElementById('learnTargetImage');
  if(!img) return;
  const label = (target || '').toUpperCase();
  if(!label){
    img.src = '';
    return;
  }
  img.src = `/assets/signs/${label}1.jpg`;
}
async function shiftLearnTarget(step){
  try{
    const r = await fetch('/api/learn-target',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({step})
    });
    const data = await r.json();
    if(!r.ok){ throw new Error(data.error || `HTTP ${r.status}`); }
    document.getElementById('target').textContent = data.target ?? '-';
    updateLearnTargetImage(data.target ?? '');
    setHint(`Target changed to ${data.target ?? '-'}.`);
  }catch(err){
    setHint(`Target switch failed: ${err}`);
  }
}
async function sendSpeech(utterance){
  const r = await fetch('/api/speech',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({utterance})
  });
  const data = await r.json();
  if(!r.ok){ throw new Error(data.error || `HTTP ${r.status}`); }
  return data;
}
function ensureSpeechRecognizer(){
  if(speechRecognizer){ return true; }
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SpeechRecognition){ return false; }
  speechRecognizer = new SpeechRecognition();
  speechRecognizer.lang = 'en-US';
  speechRecognizer.continuous = true;
  speechRecognizer.interimResults = false;
  speechRecognizer.maxAlternatives = 1;
  speechRecognizer.onresult = async (event)=>{
    const transcript = event.results[event.results.length - 1][0].transcript || '';
    const cleaned = transcript.trim();
    if(!cleaned){ return; }
    setHint(`Heard: ${cleaned}`);
    try{
      const resp = await sendSpeech(cleaned);
      document.getElementById('transcript').textContent = resp.transcript || cleaned;
      document.getElementById('letters').textContent = (resp.letters||[]).join(' ') || '-';
      renderSignGallery(resp.letters || [], resp.sign_assets || []);
    }catch(err){
      setHint(`Speech submit failed: ${err}`);
    }
  };
  speechRecognizer.onerror = (event)=>{
    setHint(`Speech error: ${event.error}`);
  };
  speechRecognizer.onend = ()=>{
    speechRunning = false;
    if(document.getElementById('currentMode').textContent.includes('speech_to_sign')){
      startSpeechRecognition();
    }
  };
  speechSupported = true;
  return true;
}
function startSpeechRecognition(){
  if(!ensureSpeechRecognizer()){
    setHint('Speech recognition is not supported in this browser.');
    return;
  }
  if(speechRunning){ return; }
  try{
    speechRecognizer.start();
    speechRunning = true;
    setHint('Listening... Speak letters or words.');
  }catch(err){
    setHint(`Speech start failed: ${err}`);
  }
}
function stopSpeechRecognition(){
  if(!speechRecognizer || !speechRunning){ return; }
  try{
    speechRecognizer.stop();
  }catch(err){
    setHint(`Speech stop failed: ${err}`);
  }
  speechRunning = false;
}
function renderSignGallery(letters, signAssets){
  const box = document.getElementById('signGallery');
  if(!box) return;
  box.innerHTML = '';
  if(!letters || letters.length === 0){
    box.innerHTML = '<div class="lbl">No recognized letters yet.</div>';
    return;
  }
  letters.forEach((letter, idx)=>{
    const asset = (signAssets && signAssets[idx]) ? signAssets[idx] : `${letter}1.jpg`;
    const src = `/assets/signs/${asset}`;
    const item = document.createElement('div');
    item.className = 'sign-item';
    item.innerHTML = `
      <img src="${src}" alt="Sign ${letter}" onerror="this.style.opacity=0.35;this.title='Missing image: ${asset}'" />
      <div class="lbl">${letter}</div>
    `;
    box.appendChild(item);
  });
}
function toggleCard(id, show){
  const el = document.getElementById(id);
  if(!el) return;
  el.classList.toggle('hidden', !show);
}
function applyModeLayout(mode){
  // Common cards always visible: status/prediction/confidence
  if(mode === 'read'){
    toggleCard('preview', true);
    toggleCard('card-prediction', true);
    toggleCard('card-confidence', true);
    toggleCard('card-sign-gallery', false);
    toggleCard('card-tts', true);
    toggleCard('card-transcript', false);
    toggleCard('card-letters', false);
    toggleCard('card-target', false);
    toggleCard('card-passed', false);
    toggleCard('card-learn-target-display', false);
    return;
  }
  if(mode === 'speech_to_sign'){
    toggleCard('preview', false);
    toggleCard('card-prediction', false);
    toggleCard('card-confidence', false);
    toggleCard('card-sign-gallery', true);
    toggleCard('card-tts', false);
    toggleCard('card-transcript', true);
    toggleCard('card-letters', true);
    toggleCard('card-target', false);
    toggleCard('card-passed', false);
    toggleCard('card-learn-target-display', false);
    return;
  }
  if(mode === 'learn'){
    toggleCard('preview', true);
    toggleCard('card-prediction', true);
    toggleCard('card-confidence', true);
    toggleCard('card-sign-gallery', false);
    toggleCard('card-tts', false);
    toggleCard('card-transcript', false);
    toggleCard('card-letters', false);
    toggleCard('card-target', true);
    toggleCard('card-passed', true);
    toggleCard('card-learn-target-display', true);
    return;
  }
  toggleCard('preview', true);
  toggleCard('card-prediction', true);
  toggleCard('card-confidence', true);
  toggleCard('card-sign-gallery', false);
  toggleCard('card-tts', true);
  toggleCard('card-transcript', true);
  toggleCard('card-letters', true);
  toggleCard('card-target', true);
  toggleCard('card-passed', true);
  toggleCard('card-learn-target-display', false);
}
function markActive(mode){
  ['read','speech_to_sign','learn'].forEach((m)=>{
    const b=document.getElementById('btn-'+m);
    if(b){ b.classList.toggle('active', m===mode); }
  });
  document.getElementById('currentMode').textContent = `Mode: ${mode}`;
  applyModeLayout(mode);
  if(mode === 'learn'){
    const preview = document.getElementById('preview');
    if(preview){
      preview.classList.remove('hidden');
      preview.src = `/video.jpg?t=${Date.now()}`;
    }
  }
  if(mode === 'speech_to_sign'){
    startSpeechRecognition();
  }else{
    stopSpeechRecognition();
  }
}
async function setMode(mode){
  const btn = document.getElementById('btn-'+mode);
  if(btn) btn.disabled = true;
  setHint(`Switching mode to ${mode}...`);
  try{
    const r = await fetch('/api/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode})});
    const data = await r.json();
    if(!r.ok){ throw new Error(data.error || `HTTP ${r.status}`); }
    markActive(data.mode || mode);
    if((data.mode || mode) !== 'speech_to_sign'){
      renderSignGallery([], []);
    }
    setHint(`Mode switched to ${data.mode || mode}.`);
  }catch(err){
    setHint(`Mode switch failed: ${err}`);
  }finally{
    if(btn) btn.disabled = false;
  }
}
async function refresh(){
  if(refreshing) return;
  refreshing = true;
  try{
    const r = await fetch('/api/state');
    const s = await r.json();
    if(!r.ok){ throw new Error(s.error || `HTTP ${r.status}`); }
    document.getElementById('status').textContent = s.status ?? '-';
    document.getElementById('prediction').textContent = s.prediction ?? '-';
    document.getElementById('confidence').textContent = `${s.confidence ?? 0}`;
    document.getElementById('tts').textContent = s.tts ?? '-';
    document.getElementById('transcript').textContent = s.transcript ?? '-';
    document.getElementById('target').textContent = s.target ?? '-';
    updateLearnTargetImage(s.target ?? '');
    document.getElementById('passed').textContent = `${s.passed ?? '-'}`;
    document.getElementById('letters').textContent = (s.letters||[]).join(' ') || '-';
    renderSignGallery(s.letters || [], s.sign_assets || []);
    if(s.mode){ markActive(s.mode); }
    if((s.mode || '') !== 'speech_to_sign'){
      const preview = document.getElementById('preview');
      preview.src = `/video.jpg?t=${Date.now()}`;
    }
  }catch(err){
    setHint(`State refresh failed: ${err}`);
  }finally{
    refreshing = false;
  }
}
setInterval(refresh, 500); refresh();
</script></body></html>"""


def build_web_server(host: str, port: int, runtime: MainRuntime, state: UIState) -> ThreadingHTTPServer:
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                html = _index_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return
            if self.path == "/api/state":
                latest_passed = bool((runtime.last_response or {}).get("passed", False))
                with lock:
                    self._send_json(
                        {
                            "mode": runtime.mode,
                            "status": state.status,
                            "prediction": "-" if runtime.latest_result is None else runtime.latest_result.label,
                            "confidence": 0.0 if runtime.latest_result is None else runtime.latest_result.confidence,
                            "transcript": runtime.latest_transcript or state.transcript,
                            "letters": state.letters,
                            "target": runtime.learn_target,
                            "passed": latest_passed if runtime.mode == "learn" else state.passed,
                            "tts": runtime.latest_tts,
                            "sign_assets": state.sign_assets,
                        }
                    )
                return
            if self.path.startswith("/assets/signs/"):
                rel = self.path.removeprefix("/assets/signs/")
                if "/" in rel or "\\" in rel:
                    self._send_json({"error": "invalid_path"}, status=400)
                    return
                assets_root = runtime.config.web.assets_dir
                asset_file = assets_root / rel
                if not asset_file.exists() or not asset_file.is_file():
                    self._send_json({"error": "not_found"}, status=404)
                    return
                data = asset_file.read_bytes()
                lower = asset_file.name.lower()
                content_type = "application/octet-stream"
                if lower.endswith(".png"):
                    content_type = "image/png"
                elif lower.endswith(".jpg") or lower.endswith(".jpeg"):
                    content_type = "image/jpeg"
                elif lower.endswith(".webp"):
                    content_type = "image/webp"
                elif lower.endswith(".gif"):
                    content_type = "image/gif"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path.startswith("/video.jpg"):
                jpg = runtime.get_latest_frame_jpeg()
                if not jpg:
                    self._send_json({"error": "no_frame_yet"}, status=503)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
                return
            self._send_json({"error": "not_found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/api/mode":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                mode = str(payload.get("mode", "read"))
                selected = runtime.set_mode(mode)
                with lock:
                    state.mode = selected
                    state.status = "active"
                self._send_json({"mode": selected})
                return

            if self.path == "/api/speech":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                utterance = str(payload.get("utterance", ""))
                result = runtime.run_speech_to_sign(utterance)
                with lock:
                    state.transcript = str(result["transcript"])
                    state.letters = list(result["letters"])
                    state.sign_assets = list(result.get("sign_assets", []))
                    state.status = "active"
                self._send_json(result)
                return

            if self.path == "/api/learn-target":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                step = int(payload.get("step", 1))
                step = 1 if step >= 0 else -1
                target = runtime.shift_learn_target(step)
                with lock:
                    state.target = target
                    state.status = "active"
                self._send_json({"target": target})
                return

            self._send_json({"error": "not_found"}, status=404)

        def log_message(self, format: str, *args) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)
