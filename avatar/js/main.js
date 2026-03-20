// js/main.js — Kiosk entry point
// Absorbs all inline scripts from index3.html:
//   - Three.js VRM renderer (portrait kiosk canvas)
//   - Clock / session / typewriter / processQuery / ask
//   - Quick-action buttons, language buttons, wake pill events
//   - WebSocket + AudioManager + SignalHandler wiring
//   - DEV: FBX drag-load + retargeting

import * as THREE from 'three';
import { GLTFLoader }              from 'three/addons/loaders/GLTFLoader.js';
import { FBXLoader }               from 'three/addons/loaders/FBXLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

import { VRMController }       from './vrm.js';
import { AnimationController } from './animation.js';
import { SecureWebSocket }     from './websocket.js';
import { SignalHandler }       from './signal.js';
import { AudioManager }        from './audio.js';

// ─────────────────────────────────────────────────────────────────────────────
// THREE.JS — Kiosk VRM renderer
// Uses the existing #vrm-canvas already in the DOM (no new canvas appended).
// ─────────────────────────────────────────────────────────────────────────────
const canvas  = document.getElementById('vrm-canvas');
const col     = canvas.parentElement;
const loading = document.getElementById('vrm-loading');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene  = new THREE.Scene();
const clock  = new THREE.Clock();

const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 20);
camera.position.set(0, 1.0, 2.0);
camera.lookAt(0, 0.85, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
keyLight.position.set(1, 2, 2);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0xa347e5, 0.6);
rimLight.position.set(-2, 1, -1);
scene.add(rimLight);

function resizeRenderer() {
  const w = col.clientWidth;
  const h = col.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
resizeRenderer();
new ResizeObserver(resizeRenderer).observe(col);

// ─────────────────────────────────────────────────────────────────────────────
// VRM + Animation controllers
// ─────────────────────────────────────────────────────────────────────────────
const vrmCtrl  = new VRMController(scene);
const animCtrl = new AnimationController(vrmCtrl);

// Load model, then hide spinner
vrmCtrl.load('./bsu_girl.vrm')
  .then(vrm => {
    vrm.scene.scale.setScalar(0.75);
    loading.style.display = 'none';
    window._devVrm = vrm; // exposed for DEV FBX loader below
    console.log('[VRM] Loaded. Expressions:', Object.keys(vrm.expressionManager?.expressionMap ?? {}));
  })
  .catch(err => {
    console.error('[VRM] Load error:', err);
    loading.querySelector('.vrm-loading-text').textContent = 'Model unavailable';
    loading.querySelector('.vrm-spinner').style.display = 'none';
  });

// Idle breathing — gently bobs chest/spine so the model doesn't look static
function applyIdle(t) {
  if (!vrmCtrl.vrm?.humanoid) return;
  const chest = vrmCtrl.vrm.humanoid.getNormalizedBoneNode('chest');
  const spine = vrmCtrl.vrm.humanoid.getNormalizedBoneNode('spine');
  if (chest) chest.rotation.x = Math.sin(t * 0.8) * 0.012;
  if (spine) spine.rotation.x = Math.sin(t * 0.8 + 0.3) * 0.008;
}

// ─────────────────────────────────────────────────────────────────────────────
// WebSocket + AudioManager + SignalHandler
// ─────────────────────────────────────────────────────────────────────────────
const signalHandler = new SignalHandler(vrmCtrl, animCtrl);

// AudioManager auto-requests mic — no button needed on a kiosk
// --- Replace your old audioMgr block with this ---

const audioMgr = new AudioManager(
  // Send raw binary PCM chunks directly to Python
  buffer => {
    if (ws) {
      ws.sendBinary(buffer);
    }
  },
  state => {
    window.dispatchEvent(new CustomEvent('iris:audiostate', { detail: state }));
  }
);
audioMgr.onMouth(v => vrmCtrl.setMouth(v));
signalHandler.audioMgr = audioMgr;

let ws;
try {
  ws = new SecureWebSocket(
    'ws://localhost:8080',
    payload => signalHandler.handle(payload),
    state   => {
      const labels = {
        connected:    'Connected to Python',
        disconnected: 'Disconnected — retrying…',
        reconnecting: 'Reconnecting…',
      };
      console.log('[WS]', labels[state] ?? state);
    },
    buf => audioMgr.receiveAudio(buf)
  );
} catch (e) {
  console.error('[Main] WebSocket init failed:', e);
}

// ─────────────────────────────────────────────────────────────────────────────
// UI — clock, session, typewriter, processQuery, quick buttons, lang, wake pill
// ─────────────────────────────────────────────────────────────────────────────

// Clock & date
function tick() {
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  document.getElementById('date-display').textContent =
    now.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
}
tick();
setInterval(tick, 1000);

// Session ID
document.getElementById('session').textContent =
  'KSK-' + Math.random().toString(36).slice(2, 6).toUpperCase();

let qCount = 0;

// Typewriter — animates the AI speech bubble text
let typeTimer = null;
function typeText(text) {
  const el = document.getElementById('aiText');
  el.innerHTML = '';
  let i = 0;
  clearInterval(typeTimer);
  typeTimer = setInterval(() => {
    if (i < text.length) {
      el.innerHTML = text.slice(0, ++i) + '<span class="cursor"></span>';
    } else {
      el.innerHTML = text;
      clearInterval(typeTimer);
    }
  }, 18);
}

function setTopic(topic) {
  document.getElementById('topicTag').textContent = topic;
}

// Greeting on load
typeText("Good day! I'm your Campus AI Assistant. I can help you with enrollment and registration steps, campus directions, citizen's charter services, official document requests, scholarship inquiries, and more. How may I assist you today?");

async function ask(text) {
  await processQuery(text);
}

async function processQuery(text) {
  qCount++;
  document.getElementById('qCount').textContent = qCount;

  const ind = document.getElementById('typingInd');
  ind.style.display = 'flex';
  document.getElementById('aiText').innerHTML =
    '<span style="color:var(--text-dimmer);font-style:italic;font-size:15px">Retrieving information from campus knowledge base…</span>';

  const t = text.toLowerCase();
  if      (t.includes('enrol')    || t.includes('registr'))                   setTopic('Enrollment');
  else if (t.includes('map')      || t.includes('direct') || t.includes('where')) setTopic('Navigation');
  else if (t.includes('charter')  || t.includes('service'))                   setTopic("Citizen's Charter");
  else if (t.includes('document') || t.includes('certif') || t.includes('tor')) setTopic('Documents');
  else if (t.includes('scholar'))                                              setTopic('Scholarship');
  else if (t.includes('pay')      || t.includes('tuition') || t.includes('fee')) setTopic('Finance');
  else if (t.includes('event')    || t.includes('announc'))                   setTopic('Announcements');
  else setTopic('General');

  try {
    const res  = await fetch('/api/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ message: text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');
    ind.style.display = 'none';
    typeText(data.reply);
  } catch (e) {
    ind.style.display = 'none';
    typeText("I'm currently unable to connect to the knowledge base. Please approach the information desk or the Registrar's Office for assistance. Thank you for your patience.");
    console.error('[API]', e);
  }
}

// Quick-action buttons — wired via data-query, no inline onclick in HTML
document.querySelectorAll('.q-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const query = btn.dataset.query;
    if (query) ask(query);
  });
});

// Language buttons
document.querySelectorAll('.lang-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
});

// Wake pill — reacts to signals from SignalHandler
const wakePill   = document.getElementById('wake-pill');
const wakeText   = document.getElementById('wake-text');
const wakeStatus = document.getElementById('wake-status');

window.addEventListener('iris:wakeword', () => {
  wakePill.classList.add('active');
  wakeStatus.textContent = 'Detected!';
  setTimeout(() => { wakeStatus.textContent = 'Listening'; }, 1000);
});

window.addEventListener('iris:listening', e => {
  if (e.detail) {
    wakePill.classList.add('active');
    wakeText.innerHTML = 'Listening…';
    wakeStatus.textContent = 'Active';
  } else {
    wakePill.classList.remove('active');
    wakeText.innerHTML = 'Say <strong>"Hey Iris"</strong> to start speaking';
    wakeStatus.textContent = 'Listening';
  }
});

window.addEventListener('iris:audiostate', e => {
  switch (e.detail) {
    case 'speaking':
      wakeStatus.textContent = 'Speaking';
      break;
    case 'listening':
      wakeStatus.textContent = 'Active';
      break;
    default:
      wakeStatus.textContent = 'Listening';
      wakePill.classList.remove('active');
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// Render loop
// ─────────────────────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  const delta   = clock.getDelta();
  const elapsed = clock.elapsedTime;
  applyIdle(elapsed);
  vrmCtrl.update(delta);
  animCtrl.update(delta);
  window._devMixerUpdate?.(delta); // DEV FBX mixer hook
  renderer.render(scene, camera);
}
animate();

// ─────────────────────────────────────────────────────────────────────────────
// DEV — FBX loader (remove entire block before deployment)
// ─────────────────────────────────────────────────────────────────────────────
const MIXAMO_TO_VRM = {
  mixamorigHips:'hips', mixamorigSpine:'spine', mixamorigSpine1:'chest',
  mixamorigSpine2:'upperChest', mixamorigNeck:'neck', mixamorigHead:'head',
  mixamorigLeftShoulder:'leftShoulder', mixamorigLeftArm:'leftUpperArm',
  mixamorigLeftForeArm:'leftLowerArm', mixamorigLeftHand:'leftHand',
  mixamorigRightShoulder:'rightShoulder', mixamorigRightArm:'rightUpperArm',
  mixamorigRightForeArm:'rightLowerArm', mixamorigRightHand:'rightHand',
  mixamorigLeftUpLeg:'leftUpperLeg', mixamorigLeftLeg:'leftLowerLeg',
  mixamorigLeftFoot:'leftFoot', mixamorigLeftToeBase:'leftToes',
  mixamorigRightUpLeg:'rightUpperLeg', mixamorigRightLeg:'rightLowerLeg',
  mixamorigRightFoot:'rightFoot', mixamorigRightToeBase:'rightToes',
  mixamorigLeftHandThumb1:'leftThumbMetacarpal', mixamorigLeftHandThumb2:'leftThumbProximal',
  mixamorigLeftHandThumb3:'leftThumbDistal',
  mixamorigLeftHandIndex1:'leftIndexProximal', mixamorigLeftHandIndex2:'leftIndexIntermediate', mixamorigLeftHandIndex3:'leftIndexDistal',
  mixamorigRightHandThumb1:'rightThumbMetacarpal', mixamorigRightHandThumb2:'rightThumbProximal',
  mixamorigRightHandThumb3:'rightThumbDistal',
  mixamorigRightHandIndex1:'rightIndexProximal', mixamorigRightHandIndex2:'rightIndexIntermediate', mixamorigRightHandIndex3:'rightIndexDistal',
};
const REALLUSION_TO_VRM = {
  'CC_Base_Hip':'hips','CC_Base_Spine01':'spine','CC_Base_Spine02':'chest',
  'CC_Base_NeckTwist01':'neck','CC_Base_Head':'head',
  'CC_Base_L_Clavicle':'leftShoulder','CC_Base_L_Upperarm':'leftUpperArm',
  'CC_Base_L_Forearm':'leftLowerArm','CC_Base_L_Hand':'leftHand',
  'CC_Base_R_Clavicle':'rightShoulder','CC_Base_R_Upperarm':'rightUpperArm',
  'CC_Base_R_Forearm':'rightLowerArm','CC_Base_R_Hand':'rightHand',
  'CC_Base_L_Thigh':'leftUpperLeg','CC_Base_L_Calf':'leftLowerLeg',
  'CC_Base_L_Foot':'leftFoot','CC_Base_L_ToeBase':'leftToes',
  'CC_Base_R_Thigh':'rightUpperLeg','CC_Base_R_Calf':'rightLowerLeg',
  'CC_Base_R_Foot':'rightFoot','CC_Base_R_ToeBase':'rightToes',
};
const DEV_BONE_MAP = { ...MIXAMO_TO_VRM, ...REALLUSION_TO_VRM };

let devMixer = null;

function devRetarget(clip, fbx, vrm) {
  const tracks = [];
  let mapped = 0, skipped = 0;

  clip.tracks.forEach(track => {
    const dot      = track.name.lastIndexOf('.');
    const boneName = track.name.slice(0, dot);
    const prop     = track.name.slice(dot + 1);
    const clean    = boneName.split('|').pop().trim();
    const vrmBone  = DEV_BONE_MAP[clean];
    if (!vrmBone) { skipped++; return; }

    const vrmNode    = vrm.humanoid.getNormalizedBoneNode(vrmBone);
    const mixamoNode = fbx.getObjectByName(clean);
    if (!vrmNode || !mixamoNode) { skipped++; return; }

    if (prop === 'quaternion') {
      const restQuatInv    = mixamoNode.quaternion.clone().invert();
      const parentWorld    = new THREE.Quaternion();
      if (mixamoNode.parent) mixamoNode.parent.getWorldQuaternion(parentWorld);
      const parentWorldInv = parentWorld.clone().invert();
      const values    = track.values;
      const newValues = new Float32Array(values.length);
      const q         = new THREE.Quaternion();
      for (let i = 0; i < values.length; i += 4) {
        q.fromArray(values, i);
        q.multiply(restQuatInv);
        q.premultiply(parentWorld);
        q.multiply(parentWorldInv);
        q.toArray(newValues, i);
      }
      tracks.push(new THREE.QuaternionKeyframeTrack(`${vrmNode.name}.${prop}`, track.times, newValues));
      mapped++;
    } else if (prop === 'position' && vrmBone === 'hips') {
      const restPosInv  = mixamoNode.position.clone().multiplyScalar(-1);
      const parentWorld = new THREE.Quaternion();
      if (mixamoNode.parent) mixamoNode.parent.getWorldQuaternion(parentWorld);
      const vrmRestPos  = vrmNode.position.clone();
      const values    = track.values;
      const newValues = new Float32Array(values.length);
      const v         = new THREE.Vector3();
      for (let i = 0; i < values.length; i += 3) {
        v.fromArray(values, i);
        v.add(restPosInv);
        v.applyQuaternion(parentWorld);
        v.multiplyScalar(0.01);
        v.add(vrmRestPos);
        v.toArray(newValues, i);
      }
      tracks.push(new THREE.VectorKeyframeTrack(`${vrmNode.name}.${prop}`, track.times, newValues));
      mapped++;
    } else {
      skipped++;
    }
  });

  console.log(`[DEV] Retargeted: ${mapped} mapped, ${skipped} skipped`);
  return new THREE.AnimationClip(clip.name, clip.duration, tracks);
}

document.getElementById('dev-fbx-input').addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;

  const vrm = window._devVrm;
  if (!vrm) { alert('VRM not loaded yet.'); return; }

  const url = URL.createObjectURL(file);
  new FBXLoader().load(url, fbx => {
    URL.revokeObjectURL(url);
    if (!fbx.animations.length) { alert('No animations in this FBX.'); return; }

    const clip = devRetarget(fbx.animations[0], fbx, vrm);

    if (devMixer) { devMixer.stopAllAction(); devMixer.uncacheRoot(devMixer.getRoot()); }
    devMixer = new THREE.AnimationMixer(vrm.scene);
    devMixer.clipAction(clip).play();

    window._devMixerUpdate = dt => devMixer.update(dt);
    document.getElementById('dev-anim-label').textContent = file.name;
    console.log('[DEV] Animation loaded:', fbx.animations[0].name);
  });
});
