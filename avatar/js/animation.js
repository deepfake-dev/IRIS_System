import * as THREE from 'three';

// ─────────────────────────────────────────────────────────────────────────────
//  PROCEDURAL ANIMATIONS — gender-aware keyframe definitions
//
//  Female: arms at z ±1.20 (relaxed A-pose), expressive head tilts,
//          lively hand gestures while talking, graceful thinking pose.
//  Male:   arms fully down z ±1.45, upright spine, broader slower sway,
//          restrained gestures, more grounded stance throughout.
//
//  All rotations are in RADIANS on the normalised VRM rig (XYZ Euler → Quaternion).
// ─────────────────────────────────────────────────────────────────────────────

const ANIMATIONS = {

  // ══════════════════════════════════════════════════════════════════════════
  //  IDLE — gentle breathing / micro-sway
  // ══════════════════════════════════════════════════════════════════════════

  female_idle: {
    duration: 4.0,
    loop: true,
    keyframes: [
      {
        time: 0.0,
        bones: {
          spine:         { x:  0.00, y:  0.00, z:  0.00 },
          chest:         { x:  0.02, y:  0.00, z:  0.00 },
          neck:          { x:  0.00, y:  0.00, z:  0.00 },
          head:          { x:  0.00, y:  0.00, z:  0.00 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.20 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.20 },
          leftLowerArm:  { x:  0.30, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.30, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 1.0,
        bones: {
          spine:         { x:  0.01, y:  0.00, z:  0.01 },
          chest:         { x:  0.03, y:  0.00, z:  0.01 },
          neck:          { x:  0.02, y:  0.00, z:  0.01 },
          head:          { x:  0.02, y:  0.01, z:  0.00 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.22 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.22 },
          leftLowerArm:  { x:  0.31, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.31, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 2.0,
        bones: {
          spine:         { x:  0.00, y:  0.00, z: -0.01 },
          chest:         { x:  0.02, y:  0.00, z: -0.01 },
          neck:          { x: -0.01, y:  0.00, z: -0.01 },
          head:          { x: -0.01, y: -0.01, z:  0.00 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.18 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.18 },
          leftLowerArm:  { x:  0.29, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.29, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 3.0,
        bones: {
          spine:         { x:  0.01, y:  0.00, z:  0.00 },
          chest:         { x:  0.03, y:  0.00, z:  0.00 },
          neck:          { x:  0.01, y:  0.00, z:  0.00 },
          head:          { x:  0.01, y:  0.00, z:  0.00 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.20 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.20 },
          leftLowerArm:  { x:  0.30, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.30, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 4.0,
        bones: {
          spine:         { x:  0.00, y:  0.00, z:  0.00 },
          chest:         { x:  0.02, y:  0.00, z:  0.00 },
          neck:          { x:  0.00, y:  0.00, z:  0.00 },
          head:          { x:  0.00, y:  0.00, z:  0.00 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.20 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.20 },
          leftLowerArm:  { x:  0.30, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.30, y:  0.00, z:  0.00 },
        }
      },
    ]
  },

  male_idle: {
    duration: 4.5,
    loop: true,
    keyframes: [
      {
        time: 0.0,
        bones: {
          spine:         { x:  0.00, y:  0.00, z:  0.00 },
          chest:         { x:  0.01, y:  0.00, z:  0.00 },
          neck:          { x:  0.00, y:  0.00, z:  0.00 },
          head:          { x:  0.00, y:  0.00, z:  0.00 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.45 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.45 },
          leftLowerArm:  { x:  0.10, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.10, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 1.0,
        bones: {
          spine:         { x:  0.01, y:  0.00, z:  0.00 },
          chest:         { x:  0.02, y:  0.00, z:  0.00 },
          neck:          { x:  0.01, y:  0.00, z:  0.00 },
          head:          { x:  0.01, y:  0.00, z:  0.00 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.46 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.46 },
          leftLowerArm:  { x:  0.11, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.11, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 2.25,
        bones: {
          spine:         { x:  0.00, y:  0.00, z: -0.01 },
          chest:         { x:  0.01, y:  0.00, z: -0.01 },
          neck:          { x: -0.01, y:  0.00, z:  0.00 },
          head:          { x: -0.01, y: -0.01, z:  0.00 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.44 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.44 },
          leftLowerArm:  { x:  0.09, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.09, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 3.5,
        bones: {
          spine:         { x:  0.01, y:  0.00, z:  0.01 },
          chest:         { x:  0.02, y:  0.00, z:  0.01 },
          neck:          { x:  0.01, y:  0.00, z:  0.00 },
          head:          { x:  0.01, y:  0.01, z:  0.00 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.46 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.46 },
          leftLowerArm:  { x:  0.11, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.11, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 4.5,
        bones: {
          spine:         { x:  0.00, y:  0.00, z:  0.00 },
          chest:         { x:  0.01, y:  0.00, z:  0.00 },
          neck:          { x:  0.00, y:  0.00, z:  0.00 },
          head:          { x:  0.00, y:  0.00, z:  0.00 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.45 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.45 },
          leftLowerArm:  { x:  0.10, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.10, y:  0.00, z:  0.00 },
        }
      },
    ]
  },

  // ══════════════════════════════════════════════════════════════════════════
  //  LISTENING — attentive lean forward, slight head tilt
  // ══════════════════════════════════════════════════════════════════════════

  female_listening: {
    duration: 3.0,
    loop: true,
    keyframes: [
      {
        time: 0.0,
        bones: {
          spine:         { x:  0.04, y:  0.00, z:  0.00 },
          chest:         { x:  0.05, y:  0.00, z:  0.00 },
          neck:          { x:  0.05, y:  0.00, z:  0.00 },
          head:          { x:  0.08, y:  0.05, z:  0.05 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.10 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.10 },
          leftLowerArm:  { x:  0.50, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.50, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 0.75,
        bones: {
          spine:         { x:  0.05, y:  0.00, z:  0.01 },
          chest:         { x:  0.06, y:  0.00, z:  0.01 },
          neck:          { x:  0.06, y:  0.01, z:  0.00 },
          head:          { x:  0.10, y:  0.06, z:  0.06 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.12 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.12 },
          leftLowerArm:  { x:  0.52, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.52, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 1.5,
        bones: {
          spine:         { x:  0.04, y:  0.00, z: -0.01 },
          chest:         { x:  0.05, y:  0.00, z: -0.01 },
          neck:          { x:  0.04, y: -0.01, z:  0.00 },
          head:          { x:  0.07, y:  0.04, z:  0.04 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.08 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.08 },
          leftLowerArm:  { x:  0.48, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.48, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 2.25,
        bones: {
          spine:         { x:  0.05, y:  0.00, z:  0.00 },
          chest:         { x:  0.06, y:  0.00, z:  0.00 },
          neck:          { x:  0.06, y:  0.00, z:  0.01 },
          head:          { x:  0.09, y:  0.05, z:  0.05 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.11 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.11 },
          leftLowerArm:  { x:  0.51, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.51, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 3.0,
        bones: {
          spine:         { x:  0.04, y:  0.00, z:  0.00 },
          chest:         { x:  0.05, y:  0.00, z:  0.00 },
          neck:          { x:  0.05, y:  0.00, z:  0.00 },
          head:          { x:  0.08, y:  0.05, z:  0.05 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.10 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.10 },
          leftLowerArm:  { x:  0.50, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.50, y:  0.00, z:  0.00 },
        }
      },
    ]
  },

  male_listening: {
    duration: 3.0,
    loop: true,
    keyframes: [
      {
        time: 0.0,
        bones: {
          spine:         { x:  0.03, y:  0.00, z:  0.00 },
          chest:         { x:  0.04, y:  0.00, z:  0.00 },
          neck:          { x:  0.04, y:  0.00, z:  0.00 },
          head:          { x:  0.06, y:  0.03, z:  0.02 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.35 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.35 },
          leftLowerArm:  { x:  0.20, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.20, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 0.75,
        bones: {
          spine:         { x:  0.04, y:  0.00, z:  0.01 },
          chest:         { x:  0.05, y:  0.00, z:  0.00 },
          neck:          { x:  0.05, y:  0.01, z:  0.00 },
          head:          { x:  0.08, y:  0.04, z:  0.02 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.36 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.36 },
          leftLowerArm:  { x:  0.22, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.22, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 1.5,
        bones: {
          spine:         { x:  0.03, y:  0.00, z: -0.01 },
          chest:         { x:  0.04, y:  0.00, z:  0.00 },
          neck:          { x:  0.03, y: -0.01, z:  0.00 },
          head:          { x:  0.05, y:  0.02, z:  0.02 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.34 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.34 },
          leftLowerArm:  { x:  0.18, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.18, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 2.25,
        bones: {
          spine:         { x:  0.04, y:  0.00, z:  0.00 },
          chest:         { x:  0.05, y:  0.00, z:  0.00 },
          neck:          { x:  0.05, y:  0.00, z:  0.01 },
          head:          { x:  0.07, y:  0.03, z:  0.02 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.35 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.35 },
          leftLowerArm:  { x:  0.21, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.21, y:  0.00, z:  0.00 },
        }
      },
      {
        time: 3.0,
        bones: {
          spine:         { x:  0.03, y:  0.00, z:  0.00 },
          chest:         { x:  0.04, y:  0.00, z:  0.00 },
          neck:          { x:  0.04, y:  0.00, z:  0.00 },
          head:          { x:  0.06, y:  0.03, z:  0.02 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.35 },
          rightUpperArm: { x:  0.00, y:  0.00, z: -1.35 },
          leftLowerArm:  { x:  0.20, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.20, y:  0.00, z:  0.00 },
        }
      },
    ]
  },

  // ══════════════════════════════════════════════════════════════════════════
  //  TALKING — expressive upper-body gesture loop
  // ══════════════════════════════════════════════════════════════════════════

  female_talking: {
    duration: 2.0,
    loop: true,
    keyframes: [
      {
        time: 0.0,
        bones: {
          spine:         { x:  0.03, y:  0.00, z:  0.00 },
          chest:         { x:  0.04, y:  0.00, z:  0.02 },
          neck:          { x:  0.03, y:  0.00, z:  0.00 },
          head:          { x:  0.03, y:  0.00, z:  0.00 },
          leftUpperArm:  { x:  0.10, y: -0.30, z:  0.90 },
          rightUpperArm: { x:  0.10, y:  0.30, z: -0.90 },
          leftLowerArm:  { x:  0.80, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.80, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.10, z:  0.10 },
          rightHand:     { x:  0.00, y: -0.10, z: -0.10 },
        }
      },
      {
        time: 0.5,
        bones: {
          spine:         { x:  0.04, y:  0.01, z:  0.01 },
          chest:         { x:  0.05, y:  0.01, z:  0.03 },
          neck:          { x:  0.04, y:  0.01, z:  0.00 },
          head:          { x:  0.04, y:  0.01, z:  0.00 },
          leftUpperArm:  { x:  0.12, y: -0.28, z:  0.85 },
          rightUpperArm: { x:  0.08, y:  0.32, z: -0.95 },
          leftLowerArm:  { x:  0.85, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.75, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.15, z:  0.15 },
          rightHand:     { x:  0.00, y: -0.05, z: -0.15 },
        }
      },
      {
        time: 1.0,
        bones: {
          spine:         { x:  0.03, y: -0.01, z: -0.01 },
          chest:         { x:  0.04, y: -0.01, z:  0.01 },
          neck:          { x:  0.02, y: -0.01, z:  0.00 },
          head:          { x:  0.02, y: -0.01, z:  0.00 },
          leftUpperArm:  { x:  0.08, y: -0.32, z:  0.95 },
          rightUpperArm: { x:  0.12, y:  0.28, z: -0.85 },
          leftLowerArm:  { x:  0.75, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.85, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.05, z:  0.05 },
          rightHand:     { x:  0.00, y: -0.15, z: -0.05 },
        }
      },
      {
        time: 1.5,
        bones: {
          spine:         { x:  0.04, y:  0.00, z:  0.02 },
          chest:         { x:  0.05, y:  0.00, z:  0.03 },
          neck:          { x:  0.04, y:  0.00, z:  0.01 },
          head:          { x:  0.04, y:  0.00, z:  0.01 },
          leftUpperArm:  { x:  0.11, y: -0.29, z:  0.88 },
          rightUpperArm: { x:  0.09, y:  0.31, z: -0.92 },
          leftLowerArm:  { x:  0.82, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.78, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.12, z:  0.12 },
          rightHand:     { x:  0.00, y: -0.08, z: -0.12 },
        }
      },
      {
        time: 2.0,
        bones: {
          spine:         { x:  0.03, y:  0.00, z:  0.00 },
          chest:         { x:  0.04, y:  0.00, z:  0.02 },
          neck:          { x:  0.03, y:  0.00, z:  0.00 },
          head:          { x:  0.03, y:  0.00, z:  0.00 },
          leftUpperArm:  { x:  0.10, y: -0.30, z:  0.90 },
          rightUpperArm: { x:  0.10, y:  0.30, z: -0.90 },
          leftLowerArm:  { x:  0.80, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.80, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.10, z:  0.10 },
          rightHand:     { x:  0.00, y: -0.10, z: -0.10 },
        }
      },
    ]
  },

  male_talking: {
    duration: 2.4,
    loop: true,
    keyframes: [
      {
        time: 0.0,
        bones: {
          spine:         { x:  0.02, y:  0.00, z:  0.00 },
          chest:         { x:  0.03, y:  0.00, z:  0.01 },
          neck:          { x:  0.02, y:  0.00, z:  0.00 },
          head:          { x:  0.02, y:  0.00, z:  0.00 },
          leftUpperArm:  { x:  0.05, y: -0.15, z:  1.10 },
          rightUpperArm: { x:  0.05, y:  0.15, z: -1.10 },
          leftLowerArm:  { x:  0.50, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.50, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.05, z:  0.05 },
          rightHand:     { x:  0.00, y: -0.05, z: -0.05 },
        }
      },
      {
        time: 0.6,
        bones: {
          spine:         { x:  0.03, y:  0.01, z:  0.01 },
          chest:         { x:  0.04, y:  0.01, z:  0.02 },
          neck:          { x:  0.03, y:  0.01, z:  0.00 },
          head:          { x:  0.03, y:  0.01, z:  0.00 },
          leftUpperArm:  { x:  0.07, y: -0.12, z:  1.05 },
          rightUpperArm: { x:  0.03, y:  0.18, z: -1.15 },
          leftLowerArm:  { x:  0.55, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.45, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.08, z:  0.08 },
          rightHand:     { x:  0.00, y: -0.03, z: -0.08 },
        }
      },
      {
        time: 1.2,
        bones: {
          spine:         { x:  0.02, y: -0.01, z: -0.01 },
          chest:         { x:  0.03, y: -0.01, z:  0.00 },
          neck:          { x:  0.01, y: -0.01, z:  0.00 },
          head:          { x:  0.01, y: -0.01, z:  0.00 },
          leftUpperArm:  { x:  0.03, y: -0.18, z:  1.15 },
          rightUpperArm: { x:  0.07, y:  0.12, z: -1.05 },
          leftLowerArm:  { x:  0.45, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.55, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.03, z:  0.03 },
          rightHand:     { x:  0.00, y: -0.08, z: -0.03 },
        }
      },
      {
        time: 1.8,
        bones: {
          spine:         { x:  0.03, y:  0.00, z:  0.01 },
          chest:         { x:  0.04, y:  0.00, z:  0.02 },
          neck:          { x:  0.03, y:  0.00, z:  0.01 },
          head:          { x:  0.03, y:  0.00, z:  0.01 },
          leftUpperArm:  { x:  0.06, y: -0.13, z:  1.08 },
          rightUpperArm: { x:  0.04, y:  0.17, z: -1.12 },
          leftLowerArm:  { x:  0.52, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.48, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.06, z:  0.06 },
          rightHand:     { x:  0.00, y: -0.06, z: -0.06 },
        }
      },
      {
        time: 2.4,
        bones: {
          spine:         { x:  0.02, y:  0.00, z:  0.00 },
          chest:         { x:  0.03, y:  0.00, z:  0.01 },
          neck:          { x:  0.02, y:  0.00, z:  0.00 },
          head:          { x:  0.02, y:  0.00, z:  0.00 },
          leftUpperArm:  { x:  0.05, y: -0.15, z:  1.10 },
          rightUpperArm: { x:  0.05, y:  0.15, z: -1.10 },
          leftLowerArm:  { x:  0.50, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  0.50, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.05, z:  0.05 },
          rightHand:     { x:  0.00, y: -0.05, z: -0.05 },
        }
      },
    ]
  },

  // ══════════════════════════════════════════════════════════════════════════
  //  THINKING — one hand raised to chin, gaze slightly upward
  // ══════════════════════════════════════════════════════════════════════════

  female_thinking: {
    duration: 3.0,
    loop: true,
    keyframes: [
      {
        time: 0.0,
        bones: {
          spine:         { x:  0.02, y:  0.00, z:  0.00 },
          chest:         { x:  0.03, y:  0.00, z:  0.00 },
          neck:          { x:  0.02, y:  0.00, z:  0.00 },
          head:          { x: -0.05, y:  0.08, z:  0.05 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.15 },
          rightUpperArm: { x: -0.20, y:  0.10, z: -0.60 },
          leftLowerArm:  { x:  0.30, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  1.20, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.00, z:  0.00 },
          rightHand:     { x:  0.10, y: -0.10, z: -0.05 },
        }
      },
      {
        time: 1.0,
        bones: {
          spine:         { x:  0.03, y:  0.01, z:  0.00 },
          chest:         { x:  0.04, y:  0.01, z:  0.00 },
          neck:          { x:  0.03, y:  0.02, z:  0.00 },
          head:          { x: -0.04, y:  0.10, z:  0.05 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.17 },
          rightUpperArm: { x: -0.22, y:  0.10, z: -0.58 },
          leftLowerArm:  { x:  0.31, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  1.22, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.00, z:  0.00 },
          rightHand:     { x:  0.11, y: -0.11, z: -0.05 },
        }
      },
      {
        time: 2.0,
        bones: {
          spine:         { x:  0.02, y: -0.01, z:  0.00 },
          chest:         { x:  0.03, y: -0.01, z:  0.00 },
          neck:          { x:  0.01, y:  0.01, z:  0.00 },
          head:          { x: -0.06, y:  0.06, z:  0.05 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.13 },
          rightUpperArm: { x: -0.18, y:  0.10, z: -0.62 },
          leftLowerArm:  { x:  0.29, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  1.18, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.00, z:  0.00 },
          rightHand:     { x:  0.09, y: -0.09, z: -0.05 },
        }
      },
      {
        time: 3.0,
        bones: {
          spine:         { x:  0.02, y:  0.00, z:  0.00 },
          chest:         { x:  0.03, y:  0.00, z:  0.00 },
          neck:          { x:  0.02, y:  0.00, z:  0.00 },
          head:          { x: -0.05, y:  0.08, z:  0.05 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.15 },
          rightUpperArm: { x: -0.20, y:  0.10, z: -0.60 },
          leftLowerArm:  { x:  0.30, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  1.20, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.00, z:  0.00 },
          rightHand:     { x:  0.10, y: -0.10, z: -0.05 },
        }
      },
    ]
  },

  male_thinking: {
    duration: 3.0,
    loop: true,
    keyframes: [
      {
        time: 0.0,
        bones: {
          spine:         { x:  0.02, y:  0.00, z:  0.00 },
          chest:         { x:  0.02, y:  0.00, z:  0.00 },
          neck:          { x:  0.02, y:  0.00, z:  0.00 },
          head:          { x: -0.04, y:  0.06, z:  0.03 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.40 },
          rightUpperArm: { x: -0.15, y:  0.10, z: -0.75 },
          leftLowerArm:  { x:  0.12, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  1.10, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.00, z:  0.00 },
          rightHand:     { x:  0.08, y: -0.08, z: -0.04 },
        }
      },
      {
        time: 1.0,
        bones: {
          spine:         { x:  0.03, y:  0.01, z:  0.00 },
          chest:         { x:  0.03, y:  0.01, z:  0.00 },
          neck:          { x:  0.03, y:  0.02, z:  0.00 },
          head:          { x: -0.03, y:  0.08, z:  0.03 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.42 },
          rightUpperArm: { x: -0.17, y:  0.10, z: -0.73 },
          leftLowerArm:  { x:  0.13, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  1.12, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.00, z:  0.00 },
          rightHand:     { x:  0.09, y: -0.09, z: -0.04 },
        }
      },
      {
        time: 2.0,
        bones: {
          spine:         { x:  0.01, y: -0.01, z:  0.00 },
          chest:         { x:  0.02, y: -0.01, z:  0.00 },
          neck:          { x:  0.01, y:  0.01, z:  0.00 },
          head:          { x: -0.05, y:  0.04, z:  0.03 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.38 },
          rightUpperArm: { x: -0.13, y:  0.10, z: -0.77 },
          leftLowerArm:  { x:  0.11, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  1.08, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.00, z:  0.00 },
          rightHand:     { x:  0.07, y: -0.07, z: -0.04 },
        }
      },
      {
        time: 3.0,
        bones: {
          spine:         { x:  0.02, y:  0.00, z:  0.00 },
          chest:         { x:  0.02, y:  0.00, z:  0.00 },
          neck:          { x:  0.02, y:  0.00, z:  0.00 },
          head:          { x: -0.04, y:  0.06, z:  0.03 },
          leftUpperArm:  { x:  0.00, y:  0.00, z:  1.40 },
          rightUpperArm: { x: -0.15, y:  0.10, z: -0.75 },
          leftLowerArm:  { x:  0.12, y:  0.00, z:  0.00 },
          rightLowerArm: { x:  1.10, y:  0.00, z:  0.00 },
          leftHand:      { x:  0.00, y:  0.00, z:  0.00 },
          rightHand:     { x:  0.08, y: -0.08, z: -0.04 },
        }
      },
    ]
  },

};


// ─────────────────────────────────────────────────────────────────────────────
//  Build a THREE.AnimationClip from one keyframe definition
// ─────────────────────────────────────────────────────────────────────────────

function buildClip(vrm, clipName, def) {
  const tracks    = [];
  const boneNames = new Set();
  def.keyframes.forEach(kf => Object.keys(kf.bones).forEach(b => boneNames.add(b)));

  boneNames.forEach(boneName => {
    const node = vrm.humanoid.getNormalizedBoneNode(boneName);
    if (!node) return;

    const times  = [];
    const values = [];

    def.keyframes.forEach(kf => {
      const rot = kf.bones[boneName];
      if (!rot) return;
      const euler = new THREE.Euler(rot.x ?? 0, rot.y ?? 0, rot.z ?? 0, 'XYZ');
      const q     = new THREE.Quaternion().setFromEuler(euler);
      times.push(kf.time);
      values.push(q.x, q.y, q.z, q.w);
    });

    if (times.length < 2) return;

    tracks.push(new THREE.QuaternionKeyframeTrack(
      `${node.name}.quaternion`,
      new Float32Array(times),
      new Float32Array(values)
    ));
  });

  return new THREE.AnimationClip(clipName, def.duration, tracks);
}


// ─────────────────────────────────────────────────────────────────────────────
//  AnimationController
// ─────────────────────────────────────────────────────────────────────────────

export class AnimationController {
  constructor(vrmController) {
    this.vrmCtrl       = vrmController;
    this.mixer         = null;
    this.currentAction = null;
    this._clips        = {};   // keyed by "gender_state", e.g. "female_idle"
    this._gender       = 'female';
    this._overrides    = {};
    this._overrideFadeSpeed = 5.0;
  }

  // ── Gender ────────────────────────────────────────────────────────────────

  setGender(gender) {
    this._gender = gender === 'male' ? 'male' : 'female';
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /** Rebuild all clips after a new VRM is loaded. */
  init() {
    const vrm = this.vrmCtrl.vrm;
    if (!vrm) return;

    if (this.mixer) this.mixer.stopAllAction();
    this.mixer         = new THREE.AnimationMixer(vrm.scene);
    this.currentAction = null;
    this._clips        = {};
    this._overrides    = {};

    for (const [key, def] of Object.entries(ANIMATIONS)) {
      this._clips[key] = buildClip(vrm, key, def);
    }
  }

  /** Wipe everything — call before loading a new model. */
  clearCache() {
    if (this.mixer) this.mixer.stopAllAction();
    this.mixer         = null;
    this.currentAction = null;
    this._clips        = {};
    this._overrides    = {};
  }

  // ── Playback ──────────────────────────────────────────────────────────────

  /**
   * Transition to a named state.
   * Resolves: gender-specific clip → opposite-gender fallback (safety net).
   *
   * @param {'idle'|'listening'|'talking'|'thinking'} state
   * @param {number} fadeDuration  seconds to crossfade (default 0.5)
   */
  playState(state, fadeDuration = 0.5) {
    const vrm = this.vrmCtrl.vrm;
    if (!vrm) return;

    if (!this.mixer) this.init();

    const primaryKey  = `${this._gender}_${state}`;
    const fallbackKey = `${this._gender === 'female' ? 'male' : 'female'}_${state}`;

    let clip = this._clips[primaryKey] ?? this._clips[fallbackKey] ?? null;

    // Build on-the-fly if not pre-built (e.g. clearCache race)
    if (!clip) {
      const def = ANIMATIONS[primaryKey] ?? ANIMATIONS[fallbackKey];
      if (!def) { console.warn(`[Anim] Unknown state: "${state}"`); return; }
      clip = buildClip(vrm, primaryKey, def);
      this._clips[primaryKey] = clip;
    }

    const defKey     = ANIMATIONS[primaryKey] ? primaryKey : fallbackKey;
    const def        = ANIMATIONS[defKey];
    const nextAction = this.mixer.clipAction(clip);
    nextAction.setLoop(def.loop ? THREE.LoopRepeat : THREE.LoopOnce, Infinity);
    nextAction.clampWhenFinished = !def.loop;
    nextAction.reset().play();

    if (this.currentAction && this.currentAction !== nextAction) {
      this.currentAction.crossFadeTo(nextAction, fadeDuration, true);
    }
    this.currentAction = nextAction;
  }

  // ── sendSignal override API ───────────────────────────────────────────────

  sendSignal({ bone, rotation }) {
    if (!bone || !rotation) return;
    const vrm = this.vrmCtrl.vrm;
    if (!vrm) return;

    const node = vrm.humanoid.getNormalizedBoneNode(bone);
    if (!node) { console.warn(`[Anim] sendSignal: bone "${bone}" not found`); return; }

    const euler  = new THREE.Euler(rotation.x ?? 0, rotation.y ?? 0, rotation.z ?? 0, 'XYZ');
    const target = new THREE.Quaternion().setFromEuler(euler);
    this._overrides[bone] = { node, target, weight: 0 };
  }

  clearSignal(bone) {
    if (bone) delete this._overrides[bone];
    else      this._overrides = {};
  }

  // ── Update loop ───────────────────────────────────────────────────────────

  update(delta) {
    if (this.mixer) this.mixer.update(delta);

    const lerpSpeed = this._overrideFadeSpeed * delta;
    for (const override of Object.values(this._overrides)) {
      override.weight = Math.min(1, override.weight + lerpSpeed);
      override.node.quaternion.slerp(override.target, override.weight);
    }
  }
}
