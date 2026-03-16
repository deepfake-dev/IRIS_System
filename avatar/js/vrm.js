// js/vrm.js — VRM model loading and expression/bone control
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

export class VRMController {
  constructor(scene) {
    this.scene   = scene;
    this.vrm     = null;

    this._loader = new GLTFLoader();
    this._loader.register(parser => new VRMLoaderPlugin(parser));
  }

  // ── Load ───────────────────────────────────────────────────────────────
  load(url) {
    return new Promise((resolve, reject) => {
      this._loader.load(
        url,
        gltf => {
          // Remove previous model if any
          if (this.vrm) {
            this.scene.remove(this.vrm.scene);
            VRMUtils.deepDispose(this.vrm.scene);
          }

          this.vrm = gltf.userData.vrm;
          VRMUtils.rotateVRM0(this.vrm);
          this.vrm.scene.traverse(obj => { if (obj.isMesh) obj.castShadow = true; });
          this.scene.add(this.vrm.scene);

          console.log(
            '[VRM] Loaded. Expressions:',
            Object.keys(this.vrm.expressionManager?.expressionMap ?? {})
          );
          resolve(this.vrm);
        },
        progress => {
          const pct = (100 * progress.loaded / (progress.total || 1)).toFixed(1);
          console.log(`[VRM] Loading ${pct}%`);
        },
        reject
      );
    });
  }

  // ── Update (call every frame) ──────────────────────────────────────────
  update(delta) {
    this.vrm?.update(delta);
  }

  // ── Expression ────────────────────────────────────────────────────────
  setExpression(name, intensity = 1.0) {
    if (!this.vrm?.expressionManager) return;
    this.vrm.expressionManager.setValue(name, Math.max(0, Math.min(1, intensity)));
  }

  setMouth(value) {
    this.setExpression('aa', value);
  }

  // ── Bone rotation ─────────────────────────────────────────────────────
  setBoneRotation(boneName, rotation = {}) {
    if (!this.vrm?.humanoid) return;
    const node = this.vrm.humanoid.getNormalizedBoneNode(boneName);
    if (!node) { console.warn(`[VRM] Bone '${boneName}' not found`); return; }
    if (rotation.x !== undefined) node.rotation.x = rotation.x;
    if (rotation.y !== undefined) node.rotation.y = rotation.y;
    if (rotation.z !== undefined) node.rotation.z = rotation.z;
  }

  // ── Head look-at ──────────────────────────────────────────────────────
  setLookAt(lookAt = {}) {
    if (!this.vrm?.humanoid) return;
    const head = this.vrm.humanoid.getNormalizedBoneNode('head');
    if (!head) return;
    if (lookAt.x !== undefined) head.rotation.y = -lookAt.x;
    if (lookAt.y !== undefined) head.rotation.x = -lookAt.y;
    if (lookAt.z !== undefined) head.rotation.z =  lookAt.z;
  }

  // ── Convenience getter ────────────────────────────────────────────────
  get ready() { return this.vrm !== null; }
}
