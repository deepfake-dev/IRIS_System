import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function createScene(canvas) {
  // ── Renderer ───────────────────────────────────────────────────────────
  const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  // ── Scene ──────────────────────────────────────────────────────────────
  const scene = new THREE.Scene();

  const bgLoader = new THREE.TextureLoader();
  bgLoader.load(
    './kist-1.webp',
    tex => { scene.background = tex; },
    undefined,
    ()  => { scene.background = new THREE.Color(0x1a1a2e); }
  );

  // ── Camera ─────────────────────────────────────────────────────────────
  const camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.1, 20);
  camera.position.set(0, 3, 8);

  // ── Lights ─────────────────────────────────────────────────────────────
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
  dirLight.position.set(1, 1, 1).normalize();
  dirLight.castShadow = true;
  scene.add(dirLight);
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));

  // ── Orbit controls ─────────────────────────────────────────────────────
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 3, 0); // Lowered the focus point to chest/face level
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.update();

  return { scene, camera, renderer, controls };
}