from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from time import sleep
from typing import Any

from gesturebridge.system.main_runtime import MainRuntime


def _trigger_shutdown(kill_parent: bool = True) -> None:
    """Background helper used by the /api/shutdown endpoint."""
    sleep(0.25)
    if kill_parent:
        try:
            ppid = os.getppid()
            cmdline_path = Path(f"/proc/{ppid}/cmdline")
            if cmdline_path.exists():
                raw = cmdline_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
                if "gesturebridge" in raw and "--run-daemon" in raw:
                    os.kill(ppid, signal.SIGTERM)
        except Exception:
            pass
    os._exit(0)


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
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>GestureBridge</title>
<style>
*{box-sizing:border-box}
:root{
  --bg-1:#060b16;
  --bg-2:#0b1324;
  --panel:#0f1a2f;
  --panel-soft:rgba(15,26,47,.82);
  --border:#2a3b5f;
  --text:#e6edf7;
  --muted:#9db0cf;
  --accent:#4f8cff;
  --accent-2:#7c5cff;
  --ok:#22c55e;
}
body{
  font-family:Inter,Segoe UI,Arial,sans-serif;
  margin:0;
  background:
    radial-gradient(1200px 560px at 10% -10%, rgba(79,140,255,.24), transparent 55%),
    radial-gradient(1000px 520px at 110% 0%, rgba(124,92,255,.2), transparent 50%),
    linear-gradient(120deg,var(--bg-1),var(--bg-2) 50%,#0a1220);
  color:var(--text);
  min-height:100vh;
}
.wrap{max-width:1040px;margin:20px auto 28px;padding:0 18px}
.topbar{
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:10px;
  gap:12px;
}
.title{font-size:30px;font-weight:760;letter-spacing:.2px;line-height:1.1}
.subtitle{font-size:13px;color:var(--muted);margin-top:3px;line-height:1.25}
.badge{
  font-size:13px;padding:8px 13px;border-radius:999px;
  background:linear-gradient(135deg,rgba(79,140,255,.24),rgba(124,92,255,.2));
  border:1px solid rgba(117,150,207,.45);
  color:#d8e5ff;
  box-shadow:0 8px 24px rgba(0,0,0,.22);
}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 8px;align-items:center}
button{
  padding:11px 14px;border-radius:10px;border:1px solid var(--border);
  background:linear-gradient(180deg,rgba(24,38,67,.9),rgba(17,29,53,.92));
  color:var(--text);cursor:pointer;
  transition:all .15s ease;
  font-weight:600;
  touch-action:manipulation;
  -webkit-tap-highlight-color:transparent;
}
button:hover{transform:translateY(-1px);border-color:#4d6697;background:#1a2b4e}
button.active{
  background:linear-gradient(135deg,var(--accent),var(--accent-2));
  border-color:transparent;color:#fff;
  box-shadow:0 8px 20px rgba(79,140,255,.35);
}
button.danger{
  background:linear-gradient(135deg,#e3433d,#b81c2a);
  border-color:transparent;color:#fff;
  box-shadow:0 8px 20px rgba(227,67,61,.35);
}
button.danger:hover{
  background:linear-gradient(135deg,#ef5852,#cd2533);
  border-color:transparent;
}
button:disabled{opacity:.6;cursor:wait}
.grid{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:12px;
  margin-top:14px;
}
.card{
  background:var(--panel-soft);
  backdrop-filter:blur(6px);
  border:1px solid var(--border);
  border-radius:14px;
  padding:14px;
  min-height:82px;
  box-shadow:0 10px 30px rgba(1,8,20,.28);
}
.hidden{display:none}
.stage{
  display:flex;
  flex-direction:column;
  gap:14px;
  margin-top:2px;
}
.side-panel{
  display:flex;
  flex-direction:column;
  gap:14px;
  min-width:0;
}
.side-panel .grid{margin-top:0}
.preview-shell{
  border-radius:14px;
  border:1px solid var(--border);
  background:#000;
  overflow:hidden;
  box-shadow:0 10px 30px rgba(1,8,20,.45);
  width:100%;
  max-width:min(1040px,100%);
  margin-inline:auto;
  flex-shrink:0;
  line-height:0;
}
.preview-shell .preview{
  display:block;
  margin:0 auto;
  width:auto;
  max-width:100%;
  height:auto;
  max-height:min(85dvh,calc(100vw - 36px));
  border-radius:0;
  border:none;
  box-shadow:none;
}
.learn-target-img{
  width:160px;max-width:38vw;max-height:140px;object-fit:contain;
  border-radius:10px;background:#0b1220;border:1px solid #334155;
}
.sign-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
  gap:10px;
  margin-top:12px;
}
.sign-item{
  background:var(--panel-soft);
  border:1px solid var(--border);
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
  color:#d1dff5;
}
.k{font-size:12px;color:#8fb3ff;margin-bottom:6px;text-transform:uppercase;letter-spacing:.08em}
.v{font-size:20px;font-weight:650;word-break:break-word}
.hint{
  margin-top:12px;
  color:#9ec4ff;
  font-size:13px;
  min-height:20px;
  padding:8px 10px;
  border:1px dashed rgba(121,152,212,.45);
  border-radius:10px;
  background:rgba(12,22,40,.42);
}
@media (max-width:760px){
  .wrap{padding:0 12px}
  .title{font-size:24px}
  .toolbar{gap:8px}
  button{padding:10px 12px}
  .v{font-size:18px}
}
@media (max-height:540px),(max-width:820px){
  :root{
    --stage-vh:calc(100dvh - 108px);
  }
  body{min-height:100%;overflow-x:hidden}
  .wrap{margin:6px auto 8px;padding:0 8px;max-width:100%}
  .topbar{flex-wrap:wrap;margin-bottom:4px;gap:6px}
  .title{font-size:17px}
  .subtitle{font-size:10px;margin-top:1px}
  .badge{font-size:10px;padding:4px 8px;flex-shrink:0}
  .toolbar{margin:6px 0 4px;gap:5px}
  button{padding:7px 8px;font-size:12px;border-radius:8px}
  .stage{
    flex-direction:row;
    align-items:flex-start;
    gap:8px;
    margin-top:4px;
  }
  .preview-shell{
    flex:0 1 50%;
    max-width:calc(50% - 4px);
    min-width:0;
    margin:0;
    border-radius:10px;
  }
  .preview-shell .preview{
    max-height:var(--stage-vh);
    max-width:100%;
  }
  .side-panel{
    flex:1 1 0;
    min-width:0;
    max-height:var(--stage-vh);
    overflow-x:hidden;
    overflow-y:auto;
    -webkit-overflow-scrolling:touch;
    gap:5px;
  }
  #preview-shell.hidden + .side-panel{
    max-height:none;
    align-self:stretch;
  }
  .side-panel .grid{margin-top:0}
  .grid{
    grid-template-columns:repeat(2,minmax(0,1fr));
    gap:5px;
  }
  .card{padding:6px 8px;min-height:0;border-radius:10px;box-shadow:0 4px 14px rgba(1,8,20,.22)}
  .k{font-size:9px;margin-bottom:2px}
  .v{font-size:13px;line-height:1.2}
  .hint{font-size:10px;margin-top:0;padding:5px 6px;min-height:16px;flex-shrink:0}
  .sign-grid{grid-template-columns:repeat(auto-fit,minmax(72px,1fr));gap:5px;margin-top:6px}
  .sign-item{padding:5px;border-radius:8px}
  .sign-item img{max-height:64px}
  .sign-item .lbl{font-size:10px;margin-top:3px}
  .learn-target-img{width:100px;max-width:28vw;max-height:90px}
}
</style></head><body>
<div class="wrap">
  <div class="topbar">
    <div>
      <div class="title">GestureBridge</div>
      <div class="subtitle">Real-time sign recognition and practice assistant</div>
    </div>
    <div class="badge" id="currentMode">Mode: read</div>
  </div>
  <div class="toolbar">
    <button id="btn-read" onclick="setMode('read')">Read Mode</button>
    <button id="btn-speech_to_sign" onclick="setMode('speech_to_sign')">Speech to Sign</button>
    <button id="btn-learn" onclick="setMode('learn')">Learning Practice</button>
    <button id="btn-terminate" class="danger" onclick="terminateApp()">Terminate</button>
  </div>
  <div class="stage">
    <div class="preview-shell" id="preview-shell">
      <img id="preview" class="preview" src="/video.jpg" alt="Camera preview" />
    </div>
    <div class="side-panel">
      <div class="card hidden" id="card-learn-target-display">
        <div class="k">Learning Target Sign</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap">
          <button id="btn-learn-prev" onclick="shiftLearnTarget(-1)">Prev</button>
          <img id="learnTargetImage" class="learn-target-img" src="" alt="Target sign" />
          <button id="btn-learn-next" onclick="shiftLearnTarget(1)">Next</button>
        </div>
      </div>
      <div class="grid">
        <div class="card" id="card-prediction"><div class="k">Prediction</div><div class="v" id="prediction">-</div></div>
        <div class="card" id="card-confidence"><div class="k">Confidence</div><div class="v" id="confidence">0.00</div></div>
        <div class="card" id="card-tts"><div class="k">TTS</div><div class="v" id="tts">-</div></div>
        <div class="card" id="card-transcript"><div class="k">Transcript</div><div class="v" id="transcript">-</div></div>
        <div class="card" id="card-letters"><div class="k">Letters</div><div class="v" id="letters">-</div></div>
        <div class="card" id="card-target"><div class="k">Target</div><div class="v" id="target">-</div></div>
        <div class="card" id="card-passed"><div class="k">Passed</div><div class="v" id="passed">-</div></div>
      </div>
      <div class="card hidden" id="card-vosk-record">
        <div class="k">Offline speech (Vosk)</div>
        <button type="button" id="btn-vosk-toggle" onclick="toggleVoskRecording()">Start recording</button>
        <div style="font-size:12px;color:var(--muted);margin-top:8px;line-height:1.35">
          First click starts recording from the device microphone; second click stops, recognizes speech, and shows sign images. Recording stops automatically at the maximum duration.
        </div>
      </div>
      <div class="card hidden" id="card-sign-gallery">
        <div class="k">Sign Images</div>
        <div id="signGallery" class="sign-grid"></div>
      </div>
      <div class="card hidden" id="card-word-capture">
        <div class="k">Word Recognition (WLASL-100)</div>
        <button type="button" id="btn-word-capture" onclick="startWordCapture()">Capture Word (1s)</button>
        <div id="wordStatus" style="font-size:13px;color:var(--muted);margin-top:8px"></div>
        <div id="wordPrediction" style="margin-top:10px"></div>
        <div style="font-size:12px;color:var(--muted);margin-top:6px;line-height:1.35">
          Sign a single word in front of the camera, then click Capture. We hold for 1 second and show the top-5.
        </div>
      </div>
      <div class="hint" id="hint">Ready.</div>
    </div>
  </div>
</div>
<script>
let refreshing = false;
let speechRecognizer = null;
let speechRunning = false;
let speechSupported = false;
let wordLoaded = false;
let wordCapturing = false;
let refreshTimer = null;
let terminating = false;
/** Last mode we applied via markActive (layout + speech). Avoid calling markActive every poll — it restarts Web Speech. */
let lastSyncedMode = null;
/** When true, onend must not call start() again (e.g. Chromium Web Speech "network" — cloud unreachable). */
let speechRestartSuppressed = false;
let voskRecording = false;
/** Consecutive failed /api/state polls (e.g. network unreachable). */
let statePollFailStreak = 0;
/** Delay before next poll; 500ms when healthy, backs off on errors. */
let pollIntervalMs = 500;
function setHint(msg){ document.getElementById('hint').textContent = msg; }
async function fetchStateReliable(){
  const TIMEOUT_MS = 8000;
  const ac = new AbortController();
  const tid = setTimeout(()=>ac.abort(), TIMEOUT_MS);
  try{
    return await fetch('/api/state', {cache: 'no-store', signal: ac.signal});
  }finally{
    clearTimeout(tid);
  }
}
function updateVoskButton(){
  const b = document.getElementById('btn-vosk-toggle');
  if(!b) return;
  b.textContent = voskRecording ? 'Stop and recognize' : 'Start recording';
  b.classList.toggle('active', voskRecording);
}
function toSignAssetName(label){
  const text = (label || '').trim();
  if(!text){ return ''; }
  const lower = text.toLowerCase();
  if(lower === 'del' || lower === 'space' || lower === 'nothing'){
    return `${lower}.jpg`;
  }
  const upper = text.toUpperCase();
  if(/^[A-Z]$/.test(upper)){
    return `${upper}.jpg`;
  }
  return `${lower}.jpg`;
}
function updateLearnTargetImage(target){
  const img = document.getElementById('learnTargetImage');
  if(!img) return;
  const asset = toSignAssetName(target);
  if(!asset){
    img.src = '';
    return;
  }
  img.src = `/assets/signs/${asset}`;
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
async function toggleVoskRecording(){
  const btn = document.getElementById('btn-vosk-toggle');
  if(btn) btn.disabled = true;
  try{
    if(!voskRecording){
      const r = await fetch('/api/speech-vosk/start',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
      const data = await r.json();
      if(!r.ok){ throw new Error(data.error || `HTTP ${r.status}`); }
      voskRecording = true;
      updateVoskButton();
      setHint('Recording… Speak, then press Stop and recognize.');
    }else{
      const r = await fetch('/api/speech-vosk/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
      const data = await r.json();
      if(!r.ok){ throw new Error(data.error || `HTTP ${r.status}`); }
      voskRecording = false;
      updateVoskButton();
      document.getElementById('transcript').textContent = data.transcript ?? '-';
      document.getElementById('letters').textContent = (data.letters||[]).join(' ') || '-';
      renderSignGallery(data.letters || [], data.sign_assets || []);
      setHint('Recognition finished.');
    }
  }catch(err){
    setHint(`Offline speech failed: ${err}`);
  }finally{
    if(btn) btn.disabled = false;
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
  speechRecognizer.continuous = false;
  speechRecognizer.interimResults = false;
  speechRecognizer.maxAlternatives = 1;
  speechRecognizer.onresult = async (event)=>{
    let combined = '';
    for(let i = event.resultIndex; i < event.results.length; i++){
      const res = event.results[i];
      if(!res.isFinal) continue;
      const t = (res[0] && res[0].transcript) ? String(res[0].transcript).trim() : '';
      if(t){ combined = combined ? `${combined} ${t}` : t; }
    }
    const cleaned = combined.trim();
    if(!cleaned){ return; }
    speechRestartSuppressed = false;
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
    if(event.error === 'network'){
      speechRestartSuppressed = true;
      setHint('Speech-to-text failed: network (Chromium cannot reach the cloud speech service). Check outbound HTTPS/DNS or try another network. Switch mode away and back to retry.');
    }else{
      setHint(`Speech error: ${event.error}`);
    }
  };
  speechRecognizer.onend = ()=>{
    speechRunning = false;
    if(!document.getElementById('currentMode').textContent.includes('speech_to_sign')){
      return;
    }
    if(speechRestartSuppressed){
      return;
    }
    startSpeechRecognition();
  };
  speechSupported = true;
  return true;
}
function startSpeechRecognition(){
  if(!ensureSpeechRecognizer()){
    setHint('Speech recognition is not supported in this browser.');
    return;
  }
  if(speechRunning){
    return;
  }
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
    const asset = (signAssets && signAssets[idx]) ? signAssets[idx] : toSignAssetName(letter);
    const lower = (asset || '').toLowerCase();
    const isVideo = lower.endsWith('.mp4') || lower.endsWith('.webm');
    const src = isVideo ? `/assets/word_clips/${asset}` : `/assets/signs/${asset}`;
    const item = document.createElement('div');
    item.className = 'sign-item';
    if(isVideo){
      item.innerHTML = `
        <video src="${src}" autoplay loop muted playsinline
               style="max-width:100%;height:auto;border-radius:8px"
               onerror="this.style.opacity=0.35;this.title='Missing clip: ${asset}'"></video>
        <div class="lbl">${letter}</div>
      `;
    } else {
      item.innerHTML = `
        <img src="${src}" alt="Sign ${letter}" onerror="this.style.opacity=0.35;this.title='Missing image: ${asset}'" />
        <div class="lbl">${letter}</div>
      `;
    }
    box.appendChild(item);
  });
}
function toggleCard(id, show){
  const el = document.getElementById(id);
  if(!el) return;
  el.classList.toggle('hidden', !show);
}
async function startWordCapture(){
  if(wordCapturing){ return; }
  try{
    const r = await fetch('/api/word/capture', {method:'POST', cache:'no-store'});
    const j = await r.json();
    if(!r.ok){
      setHint(`Word capture failed: ${j.error || r.status}`);
      return;
    }
    setHint('Capturing 1s — keep signing.');
  }catch(err){
    setHint(`Word capture failed: ${err}`);
  }
}
function applyModeLayout(mode){
  // Common cards always visible: prediction/confidence
  if(mode === 'read'){
    toggleCard('preview-shell', true);
    toggleCard('card-prediction', true);
    toggleCard('card-confidence', true);
    toggleCard('card-sign-gallery', false);
    toggleCard('card-tts', true);
    toggleCard('card-transcript', false);
    toggleCard('card-letters', false);
    toggleCard('card-target', false);
    toggleCard('card-passed', false);
    toggleCard('card-learn-target-display', false);
    toggleCard('card-vosk-record', false);
    toggleCard('card-word-capture', wordLoaded);
    return;
  }
  if(mode === 'speech_to_sign'){
    toggleCard('preview-shell', false);
    toggleCard('card-prediction', false);
    toggleCard('card-confidence', false);
    toggleCard('card-vosk-record', true);
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
    toggleCard('preview-shell', true);
    toggleCard('card-prediction', true);
    toggleCard('card-confidence', true);
    toggleCard('card-sign-gallery', false);
    toggleCard('card-tts', false);
    toggleCard('card-transcript', false);
    toggleCard('card-letters', false);
    toggleCard('card-target', true);
    toggleCard('card-passed', true);
    toggleCard('card-learn-target-display', true);
    toggleCard('card-vosk-record', false);
    return;
  }
  toggleCard('preview-shell', true);
  toggleCard('card-prediction', true);
  toggleCard('card-confidence', true);
  toggleCard('card-sign-gallery', false);
  toggleCard('card-tts', true);
  toggleCard('card-transcript', true);
  toggleCard('card-letters', true);
  toggleCard('card-target', true);
  toggleCard('card-passed', true);
  toggleCard('card-learn-target-display', false);
  toggleCard('card-vosk-record', false);
}
function markActive(mode){
  ['read','speech_to_sign','learn'].forEach((m)=>{
    const b=document.getElementById('btn-'+m);
    if(b){ b.classList.toggle('active', m===mode); }
  });
  document.getElementById('currentMode').textContent = `Mode: ${mode}`;
  applyModeLayout(mode);
  if(mode === 'learn'){
    const shell = document.getElementById('preview-shell');
    if(shell){ shell.classList.remove('hidden'); }
    const preview = document.getElementById('preview');
    if(preview){ preview.src = `/video.jpg?t=${Date.now()}`; }
  }
  if(mode === 'speech_to_sign'){
    speechRestartSuppressed = true;
    stopSpeechRecognition();
    voskRecording = false;
    updateVoskButton();
    setHint('Press Start recording to capture speech with the device microphone (offline Vosk).');
  }else{
    stopSpeechRecognition();
  }
}
async function terminateApp(){
  if(terminating){ return; }
  if(!confirm('Terminate GestureBridge? The program will exit.')){ return; }
  terminating = true;
  if(refreshTimer !== null){
    clearTimeout(refreshTimer);
    refreshTimer = null;
  }
  stopSpeechRecognition();
  document.querySelectorAll('button').forEach((b)=>{ b.disabled = true; });
  setHint('Shutting down GestureBridge...');
  try{
    await fetch('/api/shutdown',{method:'POST',headers:{'Content-Type':'application/json'}});
  }catch(err){
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
    const applied = data.mode || mode;
    markActive(applied);
    lastSyncedMode = applied;
    if(applied !== 'speech_to_sign'){
      renderSignGallery([], []);
    }
    if(applied === 'speech_to_sign'){
      setHint('Press Start recording to capture speech with the device microphone (offline Vosk).');
    }else{
      setHint(`Mode switched to ${applied}.`);
    }
  }catch(err){
    setHint(`Mode switch failed: ${err}`);
  }finally{
    if(btn) btn.disabled = false;
  }
}
async function refresh(){
  if(refreshing || terminating) return;
  refreshing = true;
  try{
    const r = await fetchStateReliable();
    const s = await r.json();
    if(!r.ok){ throw new Error(s.error || `HTTP ${r.status}`); }
    pollIntervalMs = 500;
    if(statePollFailStreak > 0){
      statePollFailStreak = 0;
      setHint('Connected.');
    }
    document.getElementById('prediction').textContent = s.prediction ?? '-';
    const conf = Number(s.confidence ?? 0);
    document.getElementById('confidence').textContent = Number.isFinite(conf) ? conf.toFixed(2) : '0.00';
    document.getElementById('tts').textContent = s.tts ?? '-';
    document.getElementById('transcript').textContent = s.transcript ?? '-';
    document.getElementById('target').textContent = s.target ?? '-';
    updateLearnTargetImage(s.target ?? '');
    document.getElementById('passed').textContent = `${s.passed ?? '-'}`;
    document.getElementById('letters').textContent = (s.letters||[]).join(' ') || '-';
    renderSignGallery(s.letters || [], s.sign_assets || []);
    if(typeof s.vosk_recording === 'boolean'){
      voskRecording = s.vosk_recording;
      updateVoskButton();
    }
    const vn = s.vosk_notification;
    if(vn){
      if(vn.ok && vn.result){
        const resp = vn.result;
        document.getElementById('transcript').textContent = resp.transcript ?? '-';
        document.getElementById('letters').textContent = (resp.letters||[]).join(' ') || '-';
        renderSignGallery(resp.letters || [], resp.sign_assets || []);
        setHint(vn.autostop ? 'Recording stopped (maximum duration). Recognition finished.' : 'Recognition finished.');
      }else if(!vn.ok){
        setHint(`Offline speech: ${vn.error || 'failed'}`);
      }
      voskRecording = false;
      updateVoskButton();
    }
    if(s.mode && s.mode !== lastSyncedMode){
      markActive(s.mode);
      lastSyncedMode = s.mode;
    }
    if((s.mode || '') !== 'speech_to_sign'){
      const preview = document.getElementById('preview');
      preview.src = `/video.jpg?t=${Date.now()}`;
    }
    // Word mode UI updates (Phase 2 IT-4).
    if(typeof s.word_loaded === 'boolean'){
      const newLoaded = s.word_loaded;
      if(newLoaded !== wordLoaded){
        wordLoaded = newLoaded;
        toggleCard('card-word-capture', wordLoaded && (s.mode || '') === 'read');
      }
    }
    wordCapturing = !!s.word_capturing;
    const wordBtn = document.getElementById('btn-word-capture');
    const wordStatus = document.getElementById('wordStatus');
    if(wordBtn){
      wordBtn.disabled = wordCapturing;
      wordBtn.textContent = wordCapturing
        ? `Capturing... (${s.word_buffer_filled||0}/${s.word_window_frames||30})`
        : 'Capture Word (1s)';
    }
    if(wordStatus && wordCapturing){
      wordStatus.textContent = `Recording landmarks: ${s.word_buffer_filled||0}/${s.word_window_frames||30} frames`;
    }
    if(s.word_prediction && Array.isArray(s.word_prediction.top)){
      const top = s.word_prediction.top;
      const status = s.word_prediction.status || 'confident';
      const thr = Number(s.word_prediction.threshold || 0);
      const top1prob = top.length ? top[0].prob : 0;
      const isAmbiguous = (status === 'ambiguous');
      // Banner: confident vs ambiguous. Confident = green check on top-1.
      // Ambiguous = amber "Did you mean...?" with a top-3 list.
      const bannerLabel = isAmbiguous ? 'Ambiguous — did you mean…?'
                                      : `${top[0]?.label ?? '?'} (top-1)`;
      const bannerColor = isAmbiguous ? '#f4a261' : '#22c55e';
      const bannerIcon  = isAmbiguous ? '⚠' : '✓';
      const banner = `<div style="display:flex;align-items:center;gap:8px;font-size:15px;color:${bannerColor};margin-bottom:6px"><span style="font-weight:700">${bannerIcon}</span><span>${bannerLabel}</span><span style="color:#9db0cf;font-size:11px;margin-left:auto">conf ${(top1prob*100).toFixed(0)}% • thr ${(thr*100).toFixed(0)}%</span></div>`;
      const limit = isAmbiguous ? 3 : 5;
      const rows = top.slice(0, limit).map((p, i)=>{
        const pct = Math.round(p.prob*100);
        const bar = '█'.repeat(Math.max(0, Math.round(p.prob*30)));
        const accent = (i === 0 && !isAmbiguous) ? '#22c55e' : '#7fa3ff';
        return `<div style="display:flex;gap:8px;align-items:baseline;font-size:14px"><b style="min-width:24px">#${i+1}</b><span style="min-width:120px">${p.label}</span><span style="color:#9db0cf;min-width:42px">${pct}%</span><span style="color:${accent}">${bar}</span></div>`;
      }).join('');
      const pred = document.getElementById('wordPrediction');
      if(pred) pred.innerHTML = banner + rows;
      if(wordStatus && !wordCapturing){
        const tag = isAmbiguous ? '⚠ ambiguous' : '✓ confident';
        wordStatus.textContent = `Last capture: ${s.word_prediction.elapsed_s}s — ${tag}`;
      }
    }
  }catch(err){
    statePollFailStreak += 1;
    pollIntervalMs = Math.min(Math.max(pollIntervalMs * 2, 1000), 8000);
    if(statePollFailStreak === 1){
      setHint('Reconnecting to GestureBridge…');
    }else if(statePollFailStreak >= 4){
      setHint(`Cannot reach GestureBridge (${err}). Is the app still running?`);
    }
  }finally{
    refreshing = false;
  }
}
function scheduleNextPoll(){
  if(terminating){ return; }
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(()=>{
    refresh().finally(()=>{
      if(!terminating){
        scheduleNextPoll();
      }
    });
  }, pollIntervalMs);
}
document.addEventListener('visibilitychange', ()=>{
  if(document.visibilityState === 'visible' && !terminating){
    pollIntervalMs = 500;
    refresh();
  }
});
refresh().finally(()=>{
  if(!terminating){
    scheduleNextPoll();
  }
});
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
            if self.path == "/api/diagnostics":
                # Subsystem health snapshot (Phase 2 IT-9). Cheap to compute,
                # safe under load — meant to be hit during debug sessions:
                #   curl -s http://127.0.0.1:8080/api/diagnostics | python3 -m json.tool
                try:
                    payload = runtime.diagnostics()
                except Exception as exc:
                    payload = {"error": str(exc)}
                self._send_json(payload)
                return

            if self.path == "/api/state":
                latest = runtime.last_response or {}
                latest_passed = bool(latest.get("passed", False))
                with lock:
                    self._send_json(
                        {
                            "mode": runtime.mode,
                            "status": state.status,
                            # Show ensemble output on the main prediction cards.
                            "prediction": str(latest.get("label", "-")),
                            "confidence": float(latest.get("confidence", 0.0)),
                            # Keep raw head outputs visible for diagnostics.
                            "stable_label": str(latest.get("stable_label", "nothing")),
                            "mobilenet_label": "-" if runtime.latest_result is None else runtime.latest_result.label,
                            "mobilenet_confidence": 0.0 if runtime.latest_result is None else runtime.latest_result.confidence,
                            "landmark_label": latest.get("landmark_label"),
                            "landmark_confidence": latest.get("landmark_confidence"),
                            "transcript": runtime.latest_transcript or state.transcript,
                            "letters": runtime.latest_speech_letters
                            if runtime.latest_speech_letters
                            else state.letters,
                            "target": runtime.learn_target,
                            "passed": latest_passed if runtime.mode == "learn" else state.passed,
                            "tts": runtime.latest_tts,
                            "sign_assets": runtime.latest_sign_assets
                            if runtime.latest_sign_assets
                            else state.sign_assets,
                            "vosk_recording": runtime.is_vosk_recording(),
                            "vosk_notification": runtime.take_vosk_notification(),
                            # Phase 2 IT-4 word mode UI hooks.
                            "word_loaded": runtime.word_classifier is not None,
                            "word_capturing": bool(latest.get("word_capturing", False)),
                            "word_buffer_filled": int(latest.get("word_buffer_filled", 0)),
                            "word_window_frames": runtime._word_window_frames,
                            "word_prediction": runtime.take_word_prediction(),
                        }
                    )
                return
            if self.path.startswith("/assets/signs/") or self.path.startswith(
                "/assets/word_clips/"
            ):
                if self.path.startswith("/assets/signs/"):
                    rel = self.path.removeprefix("/assets/signs/")
                    assets_root = runtime.config.web.assets_dir
                else:
                    rel = self.path.removeprefix("/assets/word_clips/")
                    assets_root = runtime.config.web.word_clips_dir
                if "/" in rel or "\\" in rel:
                    self._send_json({"error": "invalid_path"}, status=400)
                    return
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
                elif lower.endswith(".mp4"):
                    content_type = "video/mp4"
                elif lower.endswith(".webm"):
                    content_type = "video/webm"
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

            if self.path == "/api/word/capture":
                # Phase 2 IT-4: start a 1-second camera capture for word
                # recognition. The camera loop fills the buffer; client polls
                # GET /api/state for `word_prediction`.
                result = runtime.start_word_capture()
                if "error" in result:
                    self._send_json(result, status=409)
                else:
                    self._send_json(result)
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

            if self.path == "/api/speech-vosk/start":
                try:
                    runtime.start_vosk_recording()
                except RuntimeError as exc:
                    self._send_json({"error": str(exc)}, status=409)
                    return
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=500)
                    return
                self._send_json({"ok": True, "recording": True})
                return

            if self.path == "/api/speech-vosk/stop":
                try:
                    result = runtime.stop_vosk_recording_and_run_speech_to_sign()
                except RuntimeError as exc:
                    self._send_json({"error": str(exc)}, status=400)
                    return
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=500)
                    return
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

            if self.path == "/api/shutdown":
                with lock:
                    state.status = "shutting_down"
                self._send_json({"status": "shutting_down"})
                Thread(target=_trigger_shutdown, args=(True,), daemon=True).start()
                return

            self._send_json({"error": "not_found"}, status=404)

        def log_message(self, format: str, *args) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)
