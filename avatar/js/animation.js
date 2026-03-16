// js/animation.js — Mixamo FBX loading and retargeting onto a VRM model
import * as THREE from 'three';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';

// ── Mixamo → VRM humanoid bone name map ──────────────────────────────────────
const MIXAMO_TO_VRM = {
  mixamorigHips:             'hips',
  mixamorigSpine:            'spine',
  mixamorigSpine1:           'chest',
  mixamorigSpine2:           'upperChest',
  mixamorigNeck:             'neck',
  mixamorigHead:             'head',
  mixamorigLeftShoulder:     'leftShoulder',
  mixamorigLeftArm:          'leftUpperArm',
  mixamorigLeftForeArm:      'leftLowerArm',
  mixamorigLeftHand:         'leftHand',
  mixamorigRightShoulder:    'rightShoulder',
  mixamorigRightArm:         'rightUpperArm',
  mixamorigRightForeArm:     'rightLowerArm',
  mixamorigRightHand:        'rightHand',
  mixamorigLeftUpLeg:        'leftUpperLeg',
  mixamorigLeftLeg:          'leftLowerLeg',
  mixamorigLeftFoot:         'leftFoot',
  mixamorigLeftToeBase:      'leftToes',
  mixamorigRightUpLeg:       'rightUpperLeg',
  mixamorigRightLeg:         'rightLowerLeg',
  mixamorigRightFoot:        'rightFoot',
  mixamorigRightToeBase:     'rightToes',
  mixamorigLeftHandThumb1:   'leftThumbMetacarpal',
  mixamorigLeftHandThumb2:   'leftThumbProximal',
  mixamorigLeftHandThumb3:   'leftThumbDistal',
  mixamorigLeftHandIndex1:   'leftIndexProximal',
  mixamorigLeftHandIndex2:   'leftIndexIntermediate',
  mixamorigLeftHandIndex3:   'leftIndexDistal',
  mixamorigRightHandThumb1:  'rightThumbMetacarpal',
  mixamorigRightHandThumb2:  'rightThumbProximal',
  mixamorigRightHandThumb3:  'rightThumbDistal',
  mixamorigRightHandIndex1:  'rightIndexProximal',
  mixamorigRightHandIndex2:  'rightIndexIntermediate',
  mixamorigRightHandIndex3:  'rightIndexDistal',
};

export class AnimationController {
  constructor(vrmController) {
    this.vrmCtrl = vrmController;
    this.mixer   = null;
    this.action  = null;
    this._loader = new FBXLoader();
  }

  // ── Load FBX from a File object (drag-drop / input) ─────────────────────
  loadFromFile(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      this._loader.load(
        url,
        fbx => {
          URL.revokeObjectURL(url);
          if (!fbx.animations.length) {
            reject(new Error('No animations found in FBX file'));
            return;
          }
          const clip = this._retarget(fbx.animations[0]);
          this._applyClip(clip);
          resolve(fbx.animations[0].name || file.name);
        },
        undefined,
        err => { URL.revokeObjectURL(url); reject(err); }
      );
    });
  }

  // ── Load FBX from a URL ──────────────────────────────────────────────────
  loadFromURL(url) {
    return new Promise((resolve, reject) => {
      this._loader.load(
        url,
        fbx => {
          if (!fbx.animations.length) {
            reject(new Error('No animations found in FBX'));
            return;
          }
          const clip = this._retarget(fbx.animations[0]);
          this._applyClip(clip);
          resolve(fbx.animations[0].name || url);
        },
        undefined,
        reject
      );
    });
  }

  // ── Retarget Mixamo clip → VRM bone names ────────────────────────────────
  _retarget(clip) {
    const vrm    = this.vrmCtrl.vrm;
    const tracks = [];
    let mapped = 0, skipped = 0;

    clip.tracks.forEach(track => {
      const dotIdx   = track.name.lastIndexOf('.');
      const boneName = track.name.slice(0, dotIdx);
      const prop     = track.name.slice(dotIdx + 1);

      // Strip hierarchy prefix e.g. "Armature|mixamorigHips" → "mixamorigHips"
      const cleanBone   = boneName.split('|').pop().trim();
      const vrmBoneName = MIXAMO_TO_VRM[cleanBone];

      if (!vrmBoneName) { skipped++; return; }

      const vrmNode = vrm.humanoid.getNormalizedBoneNode(vrmBoneName);
      if (!vrmNode)  { skipped++; return; }

      // Drop hip position tracks to avoid floating/sinking
      // Remove this guard if you want root motion
      if (cleanBone === 'mixamorigHips' && prop === 'position') { skipped++; return; }

      const t   = track.clone();
      t.name    = `${vrmNode.name}.${prop}`;
      tracks.push(t);
      mapped++;
    });

    console.log(`[Anim] Retargeted: ${mapped} tracks mapped, ${skipped} skipped`);
    return new THREE.AnimationClip(clip.name, clip.duration, tracks);
  }

  // ── Apply retargeted clip to VRM scene ──────────────────────────────────
  _applyClip(clip) {
    const vrm = this.vrmCtrl.vrm;
    if (!vrm) return;

    // Dispose old mixer
    if (this.mixer) {
      this.mixer.stopAllAction();
      this.mixer.uncacheRoot(this.mixer.getRoot());
    }

    this.mixer  = new THREE.AnimationMixer(vrm.scene);
    this.action = this.mixer.clipAction(clip);
    this.action.play();
  }

  // ── Playback controls ────────────────────────────────────────────────────
  play()  { if (this.action) { this.action.paused = false; this.action.play(); } }
  pause() { if (this.action) this.action.paused = !this.action.paused; }
  stop()  { if (this.action) { this.action.stop(); } }

  setSpeed(value) { if (this.action) this.action.timeScale = value; }

  // ── Update (call every frame) ────────────────────────────────────────────
  update(delta) {
    this.mixer?.update(delta);
  }

  get ready() { return this.action !== null; }
}
