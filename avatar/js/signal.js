export class SignalHandler {
  constructor(vrmCtrl, animCtrl, audioMgr = null) {
    this.vrmCtrl  = vrmCtrl;
    this.animCtrl = animCtrl;
    this.audioMgr = audioMgr;
    window.sendSignal = data => this.handle(data);
  }

  handle(data) {
    if (!this.vrmCtrl.ready) return;
    if (typeof data !== 'object' || data === null) return;

    if (data.wakeword === true) {
      window.dispatchEvent(new CustomEvent('iris:wakeword'));
    }

    if (data.listening !== undefined) {
      window.dispatchEvent(new CustomEvent('iris:listening', { detail: data.listening }));
    }

    if (data.user_query !== undefined) {
      if (typeof window.setUserQuery === 'function') window.setUserQuery(data.user_query);
    }

    if (data.ai_chunk !== undefined) {
      window.dispatchEvent(new CustomEvent('iris:chunk', { detail: data.ai_chunk }));
    }

    if (data.expression !== undefined) {
      const intensity = typeof data.intensity === 'number' ? data.intensity : 1.0;
      this.vrmCtrl.setExpression(data.expression, intensity);
    }

    if (data.mouth !== undefined) {
      this.vrmCtrl.setMouth(data.mouth);
    }

    if (data.bone !== undefined && data.rotation !== undefined) {
      if (typeof data.bone === 'string' && typeof data.rotation === 'object') {
        this.vrmCtrl.setBoneRotation(data.bone, data.rotation);
      }
    }

    if (data.lookAt !== undefined && typeof data.lookAt === 'object') {
      this.vrmCtrl.setLookAt(data.lookAt);
    }

    if (data.animControl !== undefined) {
      switch (data.animControl) {
        case 'play':  this.animCtrl.play();  break;
        case 'pause': this.animCtrl.pause(); break;
        case 'stop':  this.animCtrl.stop();  break;
      }
    }

    if (data.animURL !== undefined && typeof data.animURL === 'string' && data.animURL.startsWith('/')) {
      this.animCtrl.loadFromURL(data.animURL).catch(err => console.error(err));
    }

    if (data.animSpeed !== undefined) {
      const speed = parseFloat(data.animSpeed);
      if (!isNaN(speed) && speed > 0) this.animCtrl.setSpeed(speed);
    }
  }
}