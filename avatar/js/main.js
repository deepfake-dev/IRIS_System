// js/main.js — Entry point: wires scene, VRM, animation, WebSocket, UI
import * as THREE from 'three';
import { createScene }       from './scene.js';
import { VRMController }     from './vrm.js';
import { AnimationController } from './animation.js';
import { SecureWebSocket }   from './websocket.js';
import { SignalHandler }     from './signal.js';

// ── UI elements ──────────────────────────────────────────────────────────────
const statusDot  = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const dzFbx      = document.getElementById('dz-fbx');
const inputFbx   = document.getElementById('input-fbx');
const animName   = document.getElementById('anim-name');
const btnPlay    = document.getElementById('btn-play');
const btnPause   = document.getElementById('btn-pause');
const btnStop    = document.getElementById('btn-stop');
const rangeSpeed = document.getElementById('range-speed');

function setStatus(state, message) {
  statusDot.className  = `dot ${state}`;
  statusText.textContent = message;
}

// ── Init scene ───────────────────────────────────────────────────────────────
const { scene, camera, renderer, controls } = createScene();
const clock = new THREE.Clock();

// ── Init controllers ─────────────────────────────────────────────────────────
const vrmCtrl  = new VRMController(scene);
const animCtrl = new AnimationController(vrmCtrl);

// ── Load VRM model ───────────────────────────────────────────────────────────
setStatus('loading', 'Loading model…');
vrmCtrl.load('./aba.vrm')
  .then(() => setStatus('disconnected', 'Model ready — connecting…'))
  .catch(err => {
    console.error('[Main] VRM load failed:', err);
    setStatus('disconnected', 'Model load failed — check console');
  });

// ── Signal handler ───────────────────────────────────────────────────────────
const signalHandler = new SignalHandler(vrmCtrl, animCtrl);

// ── WebSocket ────────────────────────────────────────────────────────────────
let ws;
try {
  ws = new SecureWebSocket(
    'ws://localhost:8080',
    payload => signalHandler.handle(payload),
    state => {
      const labels = {
        connected:    'Connected to Python',
        disconnected: 'Disconnected — retrying…',
        reconnecting: 'Reconnecting…',
      };
      setStatus(state, labels[state] ?? state);
    }
  );
} catch (e) {
  console.error('[Main] WebSocket init failed:', e);
  setStatus('disconnected', 'WS init error — check console');
}

// ── FBX drag-drop ─────────────────────────────────────────────────────────────
dzFbx.addEventListener('dragover', e => { e.preventDefault(); dzFbx.classList.add('over'); });
dzFbx.addEventListener('dragleave', ()  => dzFbx.classList.remove('over'));
dzFbx.addEventListener('drop', e => {
  e.preventDefault();
  dzFbx.classList.remove('over');
  const file = e.dataTransfer.files[0];
  if (file) loadAnimation(file);
});
inputFbx.addEventListener('change', e => {
  if (e.target.files[0]) loadAnimation(e.target.files[0]);
});

function loadAnimation(file) {
  if (!vrmCtrl.ready) {
    alert('VRM model not loaded yet — please wait.');
    return;
  }
  dzFbx.classList.remove('loaded');
  animName.textContent = 'Loading…';

  animCtrl.loadFromFile(file)
    .then(name => {
      animName.textContent = name;
      dzFbx.classList.add('loaded');
      dzFbx.querySelector('span').textContent = file.name;
      enableAnimControls(true);
    })
    .catch(err => {
      console.error('[Main] FBX load failed:', err);
      animName.textContent = 'Load failed';
    });
}

function enableAnimControls(enabled) {
  btnPlay.disabled    = !enabled;
  btnPause.disabled   = !enabled;
  btnStop.disabled    = !enabled;
  rangeSpeed.disabled = !enabled;
}

// ── Animation playback buttons ───────────────────────────────────────────────
btnPlay.onclick  = () => animCtrl.play();
btnPause.onclick = () => animCtrl.pause();
btnStop.onclick  = () => { animCtrl.stop(); enableAnimControls(false); animName.textContent = '—'; };

rangeSpeed.oninput = () => {
  animCtrl.setSpeed(parseFloat(rangeSpeed.value));
};

// ── Render loop ───────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  controls.update();
  vrmCtrl.update(delta);
  animCtrl.update(delta);
  renderer.render(scene, camera);
}
animate();
