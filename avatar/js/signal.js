// js/signal.js — Routes incoming signal payloads to VRM / animation controllers

export class SignalHandler {
  /**
   * @param {VRMController}       vrmCtrl
   * @param {AnimationController} animCtrl
   */
  constructor(vrmCtrl, animCtrl) {
    this.vrmCtrl  = vrmCtrl;
    this.animCtrl = animCtrl;

    // Expose globally so Python or dev tools can call window.sendSignal(...)
    window.sendSignal = data => this.handle(data);
  }

  handle(data) {
    if (!this.vrmCtrl.ready) {
      console.warn('[Signal] VRM not ready — signal ignored');
      return;
    }
    if (typeof data !== 'object' || data === null) {
      console.warn('[Signal] Invalid payload type');
      return;
    }

    // ── Expression ───────────────────────────────────────────────────────
    if (data.expression !== undefined) {
      const intensity = typeof data.intensity === 'number' ? data.intensity : 1.0;
      this.vrmCtrl.setExpression(data.expression, intensity);
    }

    // ── Mouth (convenience shorthand for 'aa' expression) ────────────────
    if (data.mouth !== undefined) {
      this.vrmCtrl.setMouth(data.mouth);
    }

    // ── Bone rotation ────────────────────────────────────────────────────
    if (data.bone !== undefined && data.rotation !== undefined) {
      if (typeof data.bone !== 'string') {
        console.warn('[Signal] bone must be a string');
      } else if (typeof data.rotation !== 'object') {
        console.warn('[Signal] rotation must be an object {x,y,z}');
      } else {
        this.vrmCtrl.setBoneRotation(data.bone, data.rotation);
      }
    }

    // ── Head look-at ─────────────────────────────────────────────────────
    if (data.lookAt !== undefined) {
      if (typeof data.lookAt !== 'object') {
        console.warn('[Signal] lookAt must be an object {x,y,z}');
      } else {
        this.vrmCtrl.setLookAt(data.lookAt);
      }
    }

    // ── Animation playback controls ──────────────────────────────────────
    if (data.animControl !== undefined) {
      switch (data.animControl) {
        case 'play':  this.animCtrl.play();  break;
        case 'pause': this.animCtrl.pause(); break;
        case 'stop':  this.animCtrl.stop();  break;
        default: console.warn(`[Signal] Unknown animControl: ${data.animControl}`);
      }
    }

    // ── Load animation from URL (e.g. served from Python backend) ────────
    if (data.animURL !== undefined) {
      if (typeof data.animURL !== 'string' || !data.animURL.startsWith('/')) {
        console.warn('[Signal] animURL must be a relative path starting with /');
        return;
      }
      this.animCtrl.loadFromURL(data.animURL)
        .then(name => console.log(`[Signal] Animation loaded: ${name}`))
        .catch(err => console.error('[Signal] Animation load failed:', err));
    }

    // ── Animation speed ──────────────────────────────────────────────────
    if (data.animSpeed !== undefined) {
      const speed = parseFloat(data.animSpeed);
      if (!isNaN(speed) && speed > 0) {
        this.animCtrl.setSpeed(speed);
      }
    }
  }
}
