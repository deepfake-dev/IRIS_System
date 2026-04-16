import * as THREE from 'three';
import { createScene }         from './scene.js';
import { VRMController }       from './vrm.js';
import { AnimationController } from './animation.js';
import { SecureWebSocket }     from './websocket.js';
import { SignalHandler }       from './signal.js';
import { AudioManager }        from './audio.js';

const canvas  = document.getElementById('vrm-canvas');
const col     = canvas.parentElement;
const loading = document.getElementById('vrm-loading');

const { scene, camera, renderer } = createScene(canvas);
const clock  = new THREE.Clock();

function resizeRenderer() {
  const w = col.clientWidth;
  const h = col.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
resizeRenderer();
new ResizeObserver(resizeRenderer).observe(col);

const vrmCtrl  = new VRMController(scene);
const animCtrl = new AnimationController(vrmCtrl);

function applyDefaultPose(gender) {
  if (gender === 'female') {
    // Female: Relaxed A-pose
    vrmCtrl.setBoneRotation('leftUpperArm',  { z: -1.2 });
    vrmCtrl.setBoneRotation('rightUpperArm', { z: 1.2 });
  } else {
    // Male: Arms resting fully down at the sides (approx 77 degrees down)
    vrmCtrl.setBoneRotation('leftUpperArm',  { z: -1.45 });
    vrmCtrl.setBoneRotation('rightUpperArm', { z: 1.45 });
  }
}

// Load initial model, then hide spinner
vrmCtrl.load('./bsu_girl.vrm')
  .then(async vrm => {
    applyDefaultPose('female');
    animCtrl.setGender('female');
    animCtrl.init(); // Build mixer + clips from the freshly loaded VRM

    loading.style.display = 'none';
    animCtrl.playState('idle'); // Start breathing immediately!
    
    console.log('[VRM] Loaded. Expressions:', Object.keys(vrm.expressionManager?.expressionMap ?? {}));
  })
  .catch(err => {
    // ... catch logic ...
    console.error('[VRM] Load error:', err);
    loading.querySelector('.vrm-loading-text').textContent = 'Model unavailable';
    loading.querySelector('.vrm-spinner').style.display = 'none';
  });

const VRM_MODELS = { female: './bsu_girl.vrm', male: './bsu_boy.vrm' };
let _currentVRMGender = 'female';

window.switchVRMModel = function(gender) {
  if (gender === _currentVRMGender) return;
  _currentVRMGender = gender;
  const url = VRM_MODELS[gender] ?? VRM_MODELS.female;

  loading.style.display = 'flex';
  const loadingText = loading.querySelector('.vrm-loading-text');
  const loadingSpinner = loading.querySelector('.vrm-spinner');
  if (loadingText) loadingText.textContent = 'Switching model…';
  if (loadingSpinner) loadingSpinner.style.display = '';

  if (ws) {
    ws.send({ command: 'set_voice', gender: gender });
  }

  animCtrl.clearCache(); // Wipe old mixer BEFORE loading the new model

  vrmCtrl.load(url)
    .then(async vrm => {
      applyDefaultPose(gender);
      animCtrl.setGender(gender);
      animCtrl.init(); // Build mixer + clips from the newly loaded VRM

      loading.style.display = 'none';
      animCtrl.playState('idle');

      console.log(`[VRM] Switched to ${gender} model.`);
    })
    .catch(err => {
      console.error('[VRM] Switch error:', err);
      if (loadingText) loadingText.textContent = 'Model unavailable';
      if (loadingSpinner) loadingSpinner.style.display = 'none';
    });
};

const signalHandler = new SignalHandler(vrmCtrl, animCtrl);

const audioMgr = new AudioManager(
  buffer => { if (ws) ws.sendBinary(buffer); },
  state => { window.dispatchEvent(new CustomEvent('iris:audiostate', { detail: state })); }
);
audioMgr.onMouth(v => vrmCtrl.setMouth(v));
signalHandler.audioMgr = audioMgr;
window._audioMgr = audioMgr;

const urlParams = new URLSearchParams(window.location.search);
const kioskLocation = urlParams.get('location') || "BatStateU Main Campus";

const locTextEl = document.getElementById('kioskLocationText');
if (locTextEl) locTextEl.textContent = kioskLocation;

let _pendingText = '';
let ws;
try {
  ws = new SecureWebSocket(
    `wss://${window.location.hostname}:7040`,
    payload => {
      // Intercept the new synced text command from Python
      if (payload.ai_text_sync !== undefined) {
        _pendingText = payload.ai_text_sync;
      } else {
        signalHandler.handle(payload);
      }
    },
    state   => {
      const banner = document.getElementById('ws-status-banner');
      const textEl = document.getElementById('ws-status-text');
      
      if (state === 'connected') {
        if (banner) banner.style.display = 'none';
        console.log('[WS] Connected to Python');

        ws.send({ command: 'init_kiosk', location: kioskLocation });
      } else {
        if (banner) banner.style.display = 'flex';
        if (textEl) {
          textEl.textContent = state === 'reconnecting' 
            ? 'Connection lost. Reconnecting...' 
            : 'Server Offline';
        }
        console.warn(`[WS] ${state}`);
      }
    },
    buf => {
      // Bundle the binary audio with the text we just received!
      audioMgr.receiveAudio(buf, _pendingText);
      _pendingText = ''; // Clear it out for the next chunk
    }
  );
} catch (e) {
  console.error('[Main] WebSocket init failed:', e);
}

const el = id => document.getElementById(id);

function setAvatarState(state) {
  const ring  = el('avatarListenRing');
  const label = el('avatarListenLabel');

  if (state === 'listening') {
    if (ring)  ring.classList.add('active');
    if (label) label.classList.add('active');
  } else {
    if (ring)  ring.classList.remove('active');
    if (label) label.classList.remove('active');
  }
}

function showUserQuery(text) {
  const bubble   = el('userBubble');
  const userText = el('userText');
  if (!bubble || !userText) return;
  userText.textContent = text.trim();
  bubble.style.display = 'block';
  const prompt = el('wakePrompt');
  if (prompt) prompt.style.display = 'none';
}

function showListeningIndicator(active) {
  const aiEl  = el('aiText');
  setAvatarState(active ? 'listening' : 'idle');

  if (active) {
    if (aiEl && !aiEl.dataset.hasResponse) {
      aiEl.innerHTML =
        '<div style="display:flex;align-items:center;gap:10px;padding:4px 0;opacity:.7;">' +
        '<span style="font-style:italic;font-size:14px;letter-spacing:1px;">Listening</span>' +
        '<span style="display:flex;gap:4px;align-items:center;">' +
        '<span class="l-dot"></span><span class="l-dot"></span><span class="l-dot"></span>' +
        '</span></div>';
    }
  } else {
    if (aiEl && !aiEl.dataset.hasResponse) aiEl.innerHTML = '';
  }
}

let _typeTimer = null;
let _typeQueue = '';
let _isTyping = false;
let _fullTypeText = '';

// =========================================================
// 1. THE DEDICATED TYPEWRITER ENGINE
// =========================================================
class TypewriterEngine {
  constructor(elementId) {
    this.containerId = elementId;
    this.queue = '';
    this.fullText = '';
    this.timer = null;
    this.isTyping = false;
  }

  startNewResponse() {
    // 1. Wipe all previous timers and memory
    if (this.timer) clearInterval(this.timer);
    this.queue = '';
    this.fullText = '';
    this.isTyping = false;
    
    // 2. Prepare the HTML containers safely
    const el = document.getElementById(this.containerId);
    if (el) {
      el.dataset.hasResponse = '1';
      // Separate the text content from the blinking cursor
      el.innerHTML = '<span class="ai-text-content"></span><span class="cursor"></span>';
    }
    _showSkipBtn(true);
  }

  append(text) {
    this.fullText += text;
    this.queue += text;
    
    if (!this.isTyping) {
      this.isTyping = true;
      this.timer = setInterval(() => {
        const el = document.getElementById(this.containerId);
        const textSpan = el ? el.querySelector('.ai-text-content') : null;
        
        // Type the next character
        if (this.queue.length > 0 && textSpan) {
          textSpan.textContent += this.queue[0];
          this.queue = this.queue.slice(1);
          el.scrollTop = el.scrollHeight;
        } else {
          // Pause if we are waiting for Python to send the next chunk
          clearInterval(this.timer);
          this.isTyping = false;
          
          if (this.queue.length === 0) {
            _showSkipBtn(false);
          }
        }
      }, 18); // Typing speed
    }
  }

  skip() {
    if (this.timer) clearInterval(this.timer);
    this.queue = '';
    this.isTyping = false;
    
    const el = document.getElementById(this.containerId);
    if (el && this.fullText) {
      // Dump the full text instantly, removing the cursor
      el.innerHTML = this.fullText;
      el.scrollTop = el.scrollHeight;
    }
    _showSkipBtn(false);
  }
}

// Initialize the global engine
const aiTypewriter = new TypewriterEngine('aiText');
aiTypewriter.startNewResponse();
aiTypewriter.append("Good day! I'm Iris, the AI Assistant for Batangas State University - TNEU - Alangilan Campus. How may I assist you today?");

// =========================================================
// 2. UI CONTROLS
// =========================================================
function _showSkipBtn(show) {
  const row = el('skipResponseRow');
  if (row) row.style.display = show ? 'flex' : 'none';
}

window.doSkipResponse = function() {
  aiTypewriter.skip();
  if (window._audioMgr) window._audioMgr.stopAll();
  if (ws) ws.send({ command: 'interrupt' });
};

window.triggerManualListen = function() {
  // 1. If the microphone isn't enabled yet, click the top button for them
  if (window._audioMgr && !window._audioMgr.isUserEnabled) {
    const micBtn = document.getElementById('enableMicBtn');
    if (micBtn) micBtn.click();
  }
  
  // 2. Tell Python to start recording instantly
  if (ws) {
    ws.send({ command: 'start_listening' });
  }
};

const wakePill   = el('wake-pill');
const wakeText   = el('wake-text');
const wakeStatus = el('wake-status');
let _responseStarted = false; // Tracks if Python has sent the first chunk yet

// =========================================================
// 3. EVENT LISTENERS
// =========================================================
window.addEventListener('iris:wakeword', () => {
  animCtrl.playState('listening');
  _responseStarted = false; // Reset state for Wake Word
  hideQR();
  
  const wp = el('wakePrompt');
  if (wp) wp.style.display = 'none';

  if (wakePill)   wakePill.classList.add('active');
  if (wakeStatus) wakeStatus.textContent = 'Detected!';
  setTimeout(() => { if (wakeStatus) wakeStatus.textContent = 'Listening'; }, 1000);
});

window.addEventListener('iris:listening', e => {
  showListeningIndicator(!!e.detail);
  const wp = el('wakePrompt');

  if (e.detail) { 
    // Recording user query
    _responseStarted = false; // Reset state for Tap-To-Speak
    animCtrl.playState('listening');
    if (wp) wp.style.display = 'none';
    if (wakePill)   wakePill.classList.add('active');
    if (wakeText)   wakeText.innerHTML = 'Listening…';
    if (wakeStatus) wakeStatus.textContent = 'Active';
  } else { 
    // Finished recording, retrieving
    animCtrl.playState('thinking');
    const ind  = el('typingInd');
    const aiEl = el('aiText');
    if (ind)  ind.style.display = 'flex';
    if (aiEl) {
      delete aiEl.dataset.hasResponse;
      aiEl.innerHTML = '<span style="font-style:italic;opacity:.6;font-size:14px">Retrieving from campus knowledge base…</span>';
    }
    if (wakePill)   wakePill.classList.remove('active');
    if (wakeText)   wakeText.innerHTML = 'Processing request...';
    if (wakeStatus) wakeStatus.textContent = 'Working';
  }
});

window.addEventListener('iris:audiostate', e => {
  const wp = el('wakePrompt');

  switch (e.detail) {
    case 'speaking':
      animCtrl.playState('talking');
      if (wakeStatus) wakeStatus.textContent = 'Speaking';
      if (wp) wp.style.display = 'none';
      showListeningIndicator(false);
      _showSkipBtn(true);
      break;

    case 'listening':
      animCtrl.playState('listening');
      if (wakeStatus) wakeStatus.textContent = 'Active';
      if (wp) wp.style.display = 'none';
      _showSkipBtn(false);
      showListeningIndicator(true);
      break;

    default: // 'idle'
      animCtrl.playState('idle');
      if (wakeStatus) wakeStatus.textContent = 'Standby';
      if (wakePill)   wakePill.classList.remove('active');
      if (wakeText) wakeText.innerHTML = 'Say <strong>"Hey Iris"</strong> to ask another question';
      
      _showSkipBtn(false);
      showListeningIndicator(false);
      
      if (wp && window._audioMgr && window._audioMgr.isUserEnabled) {
        wp.style.display = 'block';
      }
  }
});

window.addEventListener('iris:sync_text', e => {
  const ind = el('typingInd');

  // If this is the VERY FIRST chunk of a new answer, prepare the UI
  if (!_responseStarted) {
    _responseStarted = true;
    if (ind) ind.style.display = 'none';
    
    // Safely clear the screen and start the typewriter
    aiTypewriter.startNewResponse();
    
    const qCountEl = el('qCount');
    if (qCountEl) qCountEl.textContent = parseInt(qCountEl.textContent || 0) + 1;
  }

  // Pass the chunk to our new engine
  aiTypewriter.append(e.detail);

  const matches = aiTypewriter.fullText.match(URL_REGEX);
  if (matches && matches.length > 0) showQR(matches[0]);
});

const URL_REGEX = /(https?:\/\/[^\s]+)/g;
let _lastQRUrl  = null;

function showQR(url) {
  if (_lastQRUrl === url) return;
  _lastQRUrl = url;

  const panel  = el('qrPanel');
  const qrDiv  = el('qrCode');
  const qrUrl  = el('qrUrl');
  if (!panel || !qrDiv) return;

  qrDiv.innerHTML = '';
  if (typeof QRCode === 'undefined') return;
  
  new QRCode(qrDiv, {
    text:          url,
    width:         160,
    height:        160,
    colorDark:     '#191970',
    colorLight:    '#ffffff',
    correctLevel:  QRCode.CorrectLevel.M,
  });
  if (qrUrl) qrUrl.textContent = url;
  panel.style.display = 'flex';
}

function hideQR() {
  const panel = el('qrPanel');
  if (panel) panel.style.display = 'none';
  _lastQRUrl = null;
}

function animate() {
  requestAnimationFrame(animate);
  let delta = clock.getDelta();
  const elapsed = clock.elapsedTime;
  
  if (delta > 0.033) {
    delta = 0.033;
  }

  vrmCtrl.update(delta);
  animCtrl.update(delta);
  renderer.render(scene, camera);
}
animate();