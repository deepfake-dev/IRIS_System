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
  .then(vrm => {
    loading.style.display = 'none';

    applyDefaultPose('female');

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

  vrmCtrl.load(url)
    .then(vrm => {
      loading.style.display = 'none';

      applyDefaultPose(gender);

      console.log(`[VRM] Switched to ${gender} model.`);
    })
    .catch(err => {
      console.error('[VRM] Switch error:', err);
      if (loadingText) loadingText.textContent = 'Model unavailable';
      if (loadingSpinner) loadingSpinner.style.display = 'none';
    });
};

function applyIdle(t) {
  if (!vrmCtrl.vrm?.humanoid) return;
  const chest = vrmCtrl.vrm.humanoid.getNormalizedBoneNode('chest');
  const spine = vrmCtrl.vrm.humanoid.getNormalizedBoneNode('spine');
  if (chest) chest.rotation.x = Math.sin(t * 0.8) * 0.012;
  if (spine) spine.rotation.x = Math.sin(t * 0.8 + 0.3) * 0.008;
}

const signalHandler = new SignalHandler(vrmCtrl, animCtrl);

const audioMgr = new AudioManager(
  buffer => { if (ws) ws.sendBinary(buffer); },
  state => { window.dispatchEvent(new CustomEvent('iris:audiostate', { detail: state })); }
);
audioMgr.onMouth(v => vrmCtrl.setMouth(v));
signalHandler.audioMgr = audioMgr;
window._audioMgr = audioMgr;

let ws;
try {
  ws = new SecureWebSocket(
    'ws://localhost:8080',
    payload => signalHandler.handle(payload),
    state   => {
      const labels = { connected: 'Connected to Python', disconnected: 'Disconnected — retrying…', reconnecting: 'Reconnecting…' };
      console.log('[WS]', labels[state] ?? state);
    },
    buf => audioMgr.receiveAudio(buf)
  );
} catch (e) {
  console.error('[Main] WebSocket init failed:', e);
}

const el = id => document.getElementById(id);

function setAvatarState(state) {
  const ring  = el('avatarListenRing');
  const label = el('avatarListenLabel');
  const enableBtn   = el('enableMicBtn');
  const enableLabel = el('enableMicLabel');

  if (state === 'listening') {
    if (ring)  ring.classList.add('active');
    if (label) label.classList.add('active');
    if (enableBtn)   enableBtn.classList.add('active');
    if (enableLabel) enableLabel.textContent = 'Listening...';
  } else {
    if (ring)  ring.classList.remove('active');
    if (label) label.classList.remove('active');
    if (enableBtn)   enableBtn.classList.remove('active');
    if (enableLabel) enableLabel.textContent = 'Disable Microphone';
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

let typeTimer = null;
let _fullTypeText = ''; 

function _showSkipBtn(show) {
  const row = el('skipResponseRow');
  if (row) row.style.display = show ? 'flex' : 'none';
}

window.doSkipResponse = function() {
  if (typeTimer) { clearInterval(typeTimer); typeTimer = null; }
  const aiEl = el('aiText');
  if (aiEl && _fullTypeText) {
    aiEl.innerHTML = _fullTypeText;
    aiEl.scrollTop = aiEl.scrollHeight;
  }
  
  if (audioMgr) {
    audioMgr.stopAll(); // Kills active audio and empties the queue instantly
  }
  _showSkipBtn(false);

  // Shoot a message to Python to kill the generator
  if (ws) {
    ws.send({ command: 'interrupt' });
  }
};

function typeText(text) {
  const aiEl = el('aiText');
  if (!aiEl) return;
  aiEl.dataset.hasResponse = '1';
  aiEl.innerHTML = '';
  _fullTypeText = text;
  let i = 0;
  clearInterval(typeTimer);
  _showSkipBtn(true);
  typeTimer = setInterval(() => {
    if (i < text.length) {
      aiEl.innerHTML = text.slice(0, ++i) + '<span class="cursor"></span>';
    } else {
      aiEl.innerHTML = text;
      clearInterval(typeTimer);
      typeTimer = null;
      _showSkipBtn(false);
    }
  }, 18);
}

typeText("Good day! I'm Iris, the AI Assistant for Batangas State University - TNEU - Alangilan Campus. How may I assist you today?");

const wakePill   = el('wake-pill');
const wakeText   = el('wake-text');
const wakeStatus = el('wake-status');

window.addEventListener('iris:wakeword', () => {
  _responseStarted = false;
  hideQR();
  
  // Hide wake prompt when wake word is detected
  const wp = el('wakePrompt');
  if (wp) wp.style.display = 'none';

  if (wakePill)   wakePill.classList.add('active');
  if (wakeStatus) wakeStatus.textContent = 'Detected!';
  setTimeout(() => { if (wakeStatus) wakeStatus.textContent = 'Listening'; }, 1000);
});

window.addEventListener('iris:listening', e => {
  showListeningIndicator(!!e.detail);
  const wp = el('wakePrompt');

  if (e.detail) { // Recording user query
    if (wp) wp.style.display = 'none';
    if (wakePill)   wakePill.classList.add('active');
    if (wakeText)   wakeText.innerHTML = 'Listening…';
    if (wakeStatus) wakeStatus.textContent = 'Active';
  } else { // Finished recording, sending to knowledge base
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
    case 'speaking': // AI is talking out loud
      if (wakeStatus) wakeStatus.textContent = 'Speaking';
      if (wp) wp.style.display = 'none';
      showListeningIndicator(false);
      _showSkipBtn(true);
      break;

    case 'listening': // AI is recording audio
      if (wakeStatus) wakeStatus.textContent = 'Active';
      if (wp) wp.style.display = 'none';
      _showSkipBtn(false);
      showListeningIndicator(true);
      break;

    default: // 'idle' — AI finished talking, mic is waiting for wake word!
      if (wakeStatus) wakeStatus.textContent = 'Standby';
      if (wakePill)   wakePill.classList.remove('active');
      
      // Reset the top pill text
      if (wakeText) wakeText.innerHTML = 'Say <strong>"Hey Iris"</strong> to ask another question';
      
      _showSkipBtn(false);
      showListeningIndicator(false);
      
      // Bring the wake prompt back to the bottom of the chat bubble!
      if (wp && window._audioMgr && window._audioMgr._micActive) {
        wp.style.display = 'block';
      }
  }
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

let _responseStarted  = false;
let _accumulatedText  = '';

window.addEventListener('iris:chunk', e => {
  const aiEl = el('aiText');
  const ind  = el('typingInd');
  if (!aiEl) return;

  if (!_responseStarted) {
    _responseStarted = true;
    _accumulatedText = '';
    aiEl.dataset.hasResponse = '1';
    aiEl.textContent = '';
    if (ind) ind.style.display = 'none';
    
    const qCountEl = el('qCount');
    if (qCountEl) qCountEl.textContent = parseInt(qCountEl.textContent || 0) + 1;
  }

  _accumulatedText += e.detail;
  aiEl.textContent += e.detail;
  aiEl.scrollTop = aiEl.scrollHeight;

  const matches = _accumulatedText.match(URL_REGEX);
  if (matches && matches.length > 0) showQR(matches[0]);
});

function animate() {
  requestAnimationFrame(animate);
  let delta = clock.getDelta();
  const elapsed = clock.elapsedTime;
  
  if (delta > 0.033) {
    delta = 0.033;
  }

  applyIdle(elapsed);
  vrmCtrl.update(delta);
  animCtrl.update(delta);
  renderer.render(scene, camera);
}
animate();