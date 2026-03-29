export class AudioManager {
  constructor(sendBinary, onStateChange) {
    this._sendBinary    = sendBinary;
    this._onStateChange = onStateChange ?? (() => {});

    this._micCtx      = null;
    this._playbackCtx      = null;
    this._stream        = null;
    this._audioQueue    = [];
    this._isPlaying     = false;
    this._mouthCallback = null;
    this._micActive     = false;

    console.log('[Audio] Ready — call init() to request mic access.');
  }

  onMouth(cb) { this._mouthCallback = cb; }

  async init() {
    if (this._micCtx) return;   
    try {
      await this._init();
      this._onStateChange('idle');
      console.log('[Audio] Mic initialised via Enable Microphone.');
    } catch (err) {
      console.error('[Audio] Init failed:', err);
      alert('Microphone access is required. Please allow microphone permissions in your browser, then try again.');
    }
  }

  receiveAudio(arrayBuffer) {
    this._audioQueue.push(arrayBuffer);
    if (!this._isPlaying) this._playNext();
  }

  pauseMic() { this._micActive = false; }
  resumeMic() { this._micActive = true; }

  async _init() {
    if (this._micCtx) return;

    this._micCtx = new AudioContext({ sampleRate: 16000 });
    this._playbackCtx = new AudioContext(); 

    if (this._micCtx.state === 'suspended') await this._micCtx.resume();
    if (this._playbackCtx.state === 'suspended') await this._playbackCtx.resume();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("getUserMedia not supported. Are you running on HTTPS or localhost?");
    }

    this._stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });

    await this._micCtx.audioWorklet.addModule('./js/mic-processor.js');

    const source = this._micCtx.createMediaStreamSource(this._stream);
    const workletNode = new AudioWorkletNode(this._micCtx, 'mic-processor');
    
    workletNode.port.onmessage = (e) => {
      if (!this._micActive || this._isPlaying) return;
      this._sendBinary(e.data);
    };

    source.connect(workletNode);
    workletNode.connect(this._micCtx.destination);
    
    this._micActive = true;
    console.log('[Audio] Mic active. High-performance AudioWorklet streaming started.');
  }

  stopAll() {
    this._audioQueue = []; // Empty the queue
    if (this._currentSource) {
      try { this._currentSource.stop(); } catch(e) {} // Kill the active audio instantly
      this._currentSource = null;
    }
  }

  async _playNext() {
    if (this._audioQueue.length === 0) {
      this._isPlaying = false;
      this._mouthCallback?.(0);
      this._onStateChange('idle');
      this.resumeMic();
      return;
    }

    this._isPlaying = true;
    this._onStateChange('speaking');
    this.pauseMic();

    const buffer = this._audioQueue.shift();

    try {
      const audioBuffer = await this._playbackCtx.decodeAudioData(buffer.slice(0));
      const source      = this._playbackCtx.createBufferSource();
      source.buffer     = audioBuffer;

      const analyser   = this._playbackCtx.createAnalyser();
      analyser.fftSize = 256;
      const freqData   = new Uint8Array(analyser.frequencyBinCount);
      source.connect(analyser);
      analyser.connect(this._playbackCtx.destination);

      let mouthRAF;
      const animateMouth = () => {
        analyser.getByteFrequencyData(freqData);
        const avg = freqData.reduce((a, b) => a + b, 0) / freqData.length;
        this._mouthCallback?.(Math.min(avg / 80, 1.0));
        mouthRAF = requestAnimationFrame(animateMouth);
      };
      animateMouth();

      source.onended = () => {
        cancelAnimationFrame(mouthRAF);
        this._mouthCallback?.(0);
        setTimeout(() => this._playNext(), 0);
      };

      source.start();
    } catch (err) {
      console.error('[Audio] Failed to decode/play chunk:', err);
      setTimeout(() => this._playNext(), 0); 
    }
  }
}