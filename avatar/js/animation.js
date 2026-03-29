import * as THREE from 'three';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';

const MIXAMO_TO_VRM = {
  mixamorigHips: 'hips', mixamorigSpine: 'spine', mixamorigSpine1: 'chest',
  mixamorigSpine2: 'upperChest', mixamorigNeck: 'neck', mixamorigHead: 'head',
  mixamorigLeftShoulder: 'leftShoulder', mixamorigLeftArm: 'leftUpperArm',
  mixamorigLeftForeArm: 'leftLowerArm', mixamorigLeftHand: 'leftHand',
  mixamorigRightShoulder: 'rightShoulder', mixamorigRightArm: 'rightUpperArm',
  mixamorigRightForeArm: 'rightLowerArm', mixamorigRightHand: 'rightHand',
  mixamorigLeftUpLeg: 'leftUpperLeg', mixamorigLeftLeg: 'leftLowerLeg',
  mixamorigLeftFoot: 'leftFoot', mixamorigLeftToeBase: 'leftToes',
  mixamorigRightUpLeg: 'rightUpperLeg', mixamorigRightLeg: 'rightLowerLeg',
  mixamorigRightFoot: 'rightFoot', mixamorigRightToeBase: 'rightToes',
  mixamorigLeftHandThumb1: 'leftThumbMetacarpal', mixamorigLeftHandThumb2: 'leftThumbProximal', mixamorigLeftHandThumb3: 'leftThumbDistal',
  mixamorigLeftHandIndex1: 'leftIndexProximal', mixamorigLeftHandIndex2: 'leftIndexIntermediate', mixamorigLeftHandIndex3: 'leftIndexDistal',
  mixamorigRightHandThumb1: 'rightThumbMetacarpal', mixamorigRightHandThumb2: 'rightThumbProximal', mixamorigRightHandThumb3: 'rightThumbDistal',
  mixamorigRightHandIndex1: 'rightIndexProximal', mixamorigRightHandIndex2: 'rightIndexIntermediate', mixamorigRightHandIndex3: 'rightIndexDistal',
};

const REALLUSION_TO_VRM = {
  'CC_Base_Hip': 'hips', 'CC_Base_Spine01': 'spine', 'CC_Base_Spine02': 'chest',
  'CC_Base_NeckTwist01': 'neck', 'CC_Base_Head': 'head',
  'CC_Base_L_Clavicle': 'leftShoulder', 'CC_Base_L_Upperarm': 'leftUpperArm', 'CC_Base_L_Forearm': 'leftLowerArm', 'CC_Base_L_Hand': 'leftHand',
  'CC_Base_R_Clavicle': 'rightShoulder', 'CC_Base_R_Upperarm': 'rightUpperArm', 'CC_Base_R_Forearm': 'rightLowerArm', 'CC_Base_R_Hand': 'rightHand',
  'CC_Base_L_Thigh': 'leftUpperLeg', 'CC_Base_L_Calf': 'leftLowerLeg', 'CC_Base_L_Foot': 'leftFoot', 'CC_Base_L_ToeBase': 'leftToes',
  'CC_Base_R_Thigh': 'rightUpperLeg', 'CC_Base_R_Calf': 'rightLowerLeg', 'CC_Base_R_Foot': 'rightFoot', 'CC_Base_R_ToeBase': 'rightToes'
};

const BONE_MAP = { ...MIXAMO_TO_VRM, ...REALLUSION_TO_VRM };

export class AnimationController {
  constructor(vrmController) {
    this.vrmCtrl = vrmController;
    this.mixer   = null;
    this.action  = null;
    this._loader = new FBXLoader();
  }

  loadFromFile(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      this._loader.load(
        url,
        fbx => {
          URL.revokeObjectURL(url);
          if (!fbx.animations.length) { reject(new Error('No animations found in FBX file')); return; }
          const clip = this._retarget(fbx.animations[0], fbx);
          this._applyClip(clip);
          resolve(fbx.animations[0].name || file.name);
        },
        undefined,
        err => { URL.revokeObjectURL(url); reject(err); }
      );
    });
  }

  _retarget(clip, fbx) {
    const vrm    = this.vrmCtrl.vrm;
    const tracks = [];

    clip.tracks.forEach(track => {
      const dotIdx   = track.name.lastIndexOf('.');
      const boneName = track.name.slice(0, dotIdx);
      const prop     = track.name.slice(dotIdx + 1);

      const cleanBone   = boneName.split('|').pop().trim();
      const vrmBoneName = BONE_MAP[cleanBone];

      if (!vrmBoneName) return;

      const vrmNode = vrm.humanoid.getNormalizedBoneNode(vrmBoneName);
      const mixamoNode = fbx.getObjectByName(cleanBone);

      if (!vrmNode || !mixamoNode) return;

      if (prop === 'quaternion') {
        const restQuatInv = mixamoNode.quaternion.clone().invert();
        const parentWorld = new THREE.Quaternion();
        if (mixamoNode.parent) mixamoNode.parent.getWorldQuaternion(parentWorld);
        const parentWorldInv = parentWorld.clone().invert();

        const values = track.values;
        const newValues = new Float32Array(values.length);
        const q = new THREE.Quaternion();

        for (let i = 0; i < values.length; i += 4) {
          q.fromArray(values, i);
          q.multiply(restQuatInv);
          q.premultiply(parentWorld);
          q.multiply(parentWorldInv);
          q.toArray(newValues, i);
        }

        tracks.push(new THREE.QuaternionKeyframeTrack(`${vrmNode.name}.${prop}`, track.times, newValues));

      } else if (prop === 'position' && vrmBoneName === 'hips') {
        const restPosInv = mixamoNode.position.clone().multiplyScalar(-1);
        const parentWorld = new THREE.Quaternion();
        if (mixamoNode.parent) mixamoNode.parent.getWorldQuaternion(parentWorld);
        const vrmRestPos = vrmNode.position.clone();

        const values = track.values;
        const newValues = new Float32Array(values.length);
        const v = new THREE.Vector3();

        for (let i = 0; i < values.length; i += 3) {
          v.fromArray(values, i);
          v.add(restPosInv);
          v.applyQuaternion(parentWorld);
          v.multiplyScalar(0.01);
          v.add(vrmRestPos);
          v.toArray(newValues, i);
        }

        tracks.push(new THREE.VectorKeyframeTrack(`${vrmNode.name}.${prop}`, track.times, newValues));
      }
    });

    return new THREE.AnimationClip(clip.name, clip.duration, tracks);
  }

  _applyClip(clip) {
    const vrm = this.vrmCtrl.vrm;
    if (!vrm) return;
    if (this.mixer) {
      this.mixer.stopAllAction();
      this.mixer.uncacheRoot(this.mixer.getRoot());
    }
    this.mixer  = new THREE.AnimationMixer(vrm.scene);
    this.action = this.mixer.clipAction(clip);
    this.action.play();
  }

  play()  { if (this.action) { this.action.paused = false; this.action.play(); } }
  pause() { if (this.action) this.action.paused = !this.action.paused; }
  stop()  { if (this.action) { this.action.stop(); } }
  setSpeed(value) { if (this.action) this.action.timeScale = value; }

  update(delta) { this.mixer?.update(delta); }
  get ready() { return this.action !== null; }
}