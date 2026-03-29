import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

export class VRMController {
  constructor(scene) {
    this.scene   = scene;
    this.vrm     = null;
    
    // THE FIX: An invisible wrapper to hold the avatar.
    // We scale/move this wrapper so the VRM's internal physics stay at 1.0
    this.avatarGroup = new THREE.Group();
    this.scene.add(this.avatarGroup);

    this._loader = new GLTFLoader();
    this._loader.register(parser => new VRMLoaderPlugin(parser));
  }

  load(url) {
  return new Promise((resolve, reject) => {
    this._loader.load(
      url,
      gltf => {
        // === CLEANUP OLD MODEL ===
        if (this.vrm) {
          this.avatarGroup.remove(this.vrm.scene);
          // ... your existing disposal code stays here ...
          this.vrm = null;
        }

        this.vrm = gltf.userData.vrm;
        VRMUtils.rotateVRM0(this.vrm);

        this.vrm.scene.traverse(obj => {
          if (obj.isMesh) obj.castShadow = true;
        });

        // === FORCE CLEAN REST POSE ===
        if (this.vrm.humanoid) {
          this.vrm.humanoid.resetNormalizedPose();

          // Extra safety — zero every bone
          this.vrm.scene.traverse((obj) => {
            if (obj.isBone) obj.rotation.set(0, 0, 0);
          });
        }

        this.avatarGroup.add(this.vrm.scene);

        // =============================================================
        // 1. FULLY RESET WRAPPER BEFORE MEASURING (Fixes the drifting view)
        // =============================================================
        this.avatarGroup.position.set(0, 0, 0);
        this.avatarGroup.rotation.set(0, 0, 0);
        this.avatarGroup.scale.setScalar(1);
        
        // Force Three.js to calculate the zeroed-out world matrix
        this.avatarGroup.updateMatrixWorld(true);

        // =============================================================
        // 2. SKELETON-BASED SCALING
        // =============================================================
        const hips = this.vrm.humanoid?.getNormalizedBoneNode('hips');
        const head = this.vrm.humanoid?.getNormalizedBoneNode('head');

        if (hips && head) {
          const hipsWorld = new THREE.Vector3();
          const headWorld = new THREE.Vector3();
          hips.getWorldPosition(hipsWorld);
          head.getWorldPosition(headWorld);

          const currentHipsToHead = headWorld.y - hipsWorld.y;
          const targetHipsToHead  = 1.40;          

          const scaleFactor = targetHipsToHead / currentHipsToHead;
          this.avatarGroup.scale.setScalar(scaleFactor);
          
          // Force matrix update again so the bounding box accurately reads the new scale
          this.avatarGroup.updateMatrixWorld(true);

          console.log(`[VRM] scaleFactor: ${scaleFactor.toFixed(4)}`);
        }

        // =============================================================
        // 3. CENTERING (Using World Bounding Box)
        // =============================================================
        const box = new THREE.Box3().setFromObject(this.vrm.scene);
        const center = box.getCenter(new THREE.Vector3());

        // Because the box is measured in world space and the wrapper is at (0,0,0),
        // we do NOT multiply by scale here. We just subtract the pure coordinates.
        this.avatarGroup.position.x = -center.x;
        this.avatarGroup.position.z = -center.z;

        // Put feet exactly on the floor (small offset so shoes don't sink)
        this.avatarGroup.position.y = -box.min.y + 0.03;

        // Force one physics update to settle
        try {
            this.vrm.springBoneManager?.reset();
            this.vrm.update(0);
        } catch(e) {}

        resolve(this.vrm);
      },
      undefined,
      reject
    );
  });
}

  update(delta) {
    this.vrm?.update(delta);
  }

  setExpression(name, intensity = 1.0) {
    if (!this.vrm?.expressionManager) return;
    this.vrm.expressionManager.setValue(name, Math.max(0, Math.min(1, intensity)));
  }

  setMouth(value) {
    this.setExpression('aa', value);
  }

  setBoneRotation(boneName, rotation = {}) {
    if (!this.vrm?.humanoid) return;
    const node = this.vrm.humanoid.getNormalizedBoneNode(boneName);
    if (!node) { console.warn(`[VRM] Bone '${boneName}' not found`); return; }
    if (rotation.x !== undefined) node.rotation.x = rotation.x;
    if (rotation.y !== undefined) node.rotation.y = rotation.y;
    if (rotation.z !== undefined) node.rotation.z = rotation.z;
  }

  setLookAt(lookAt = {}) {
    if (!this.vrm?.humanoid) return;
    const head = this.vrm.humanoid.getNormalizedBoneNode('head');
    if (!head) return;
    if (lookAt.x !== undefined) head.rotation.y = -lookAt.x;
    if (lookAt.y !== undefined) head.rotation.x = -lookAt.y;
    if (lookAt.z !== undefined) head.rotation.z =  lookAt.z;
  }

  get ready() { return this.vrm !== null; }
}