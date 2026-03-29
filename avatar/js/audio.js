export class AudioManager {
  constructor(sendBinary, onStateChange) {
    this._sendBinary    = sendBinary;
    this._onStateChange = onStateChange ?? (() => {});

    this._micCtx        = null;
    this._playbackCtx   = null;
    this._stream        = null;
    this._audioQueue    = [];
    this._isPlaying     = false;
    this._mouthCallback = null;
    
    // AI's internal state (turns off when speaking/processing)
    this._micActive     = false;
    
    // USER'S master switch (controlled ONLY by the button)
    this.isUserEnabled  = false; 

    console.log('[Audio] Ready — call init() to request mic access.');
  }

  onMouth(cb) { this._mouthCallback = cb; }

  async init() {
    this.isUserEnabled = true; 
    
    if (this._micCtx) {
      this.enableUserMic();
      return;   
    }
    
    try {
      await this._init();
      this._onStateChange('idle');
      console.log('[Audio] Mic initialised.');
    } catch (err) {
      console.error('[Audio] Init failed:', err);
      alert('Microphone access is required.');
    }
  }

  // --- STRICT MANUAL TOGGLES ---
  disableUserMic() {
    this.isUserEnabled = false;
    console.log('[Audio] User Muted: Data stream stopped.');
  }

  enableUserMic() {
    this.isUserEnabled = true;
    console.log('[Audio] User Unmuted: Data stream active.');
  }
  // -----------------------------

  receiveAudio(arrayBuffer, textChunk = '') {
    this._audioQueue.push({ buffer: arrayBuffer, text: textChunk });
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

    this._stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });

    await this._micCtx.audioWorklet.addModule('./js/mic-processor.js');

    const source = this._micCtx.createMediaStreamSource(this._stream);
    const workletNode = new AudioWorkletNode(this._micCtx, 'mic-processor');
    
    workletNode.port.onmessage = (e) => {
      // THE GATE: Drop the audio if the user clicked "Disable", OR if AI is speaking/processing
      if (!this.isUserEnabled || !this._micActive || this._isPlaying) return;
      this._sendBinary(e.data);
    };

    source.connect(workletNode);
    workletNode.connect(this._micCtx.destination);
    
    this._micActive = true;
  }

  stopAll() {
    this._audioQueue = [];
    if (this._currentSource) {
      try { this._currentSource.stop(); } catch(e) {}
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

    const item = this._audioQueue.shift();
    const buffer = item.buffer;
    const textToDisplay = item.text;

    if (textToDisplay) {
      window.dispatchEvent(new CustomEvent('iris:sync_text', { detail: textToDisplay }));
    }

    try {
      const audioBuffer = await this._playbackCtx.decodeAudioData(buffer.slice(0));
      const source      = this._playbackCtx.createBufferSource();
      source.buffer     = audioBuffer;
      this._currentSource = source;

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
      setTimeout(() => this._playNext(), 0); 
    }
  }
}