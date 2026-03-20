// js/audio.js — Raw PCM Streaming to Python + TTS audio playback

export class AudioManager {
  constructor(sendBinary, onStateChange) {
    this._sendBinary    = sendBinary;
    this._onStateChange = onStateChange ?? (() => {});

    this._micCtx      = null;
    this._playbackCtx      = null;
    this._stream        = null;
    this._processor     = null;
    this._audioQueue    = [];
    this._isPlaying     = false;
    this._mouthCallback = null;
    this._micActive     = false;

    // Must wait for user interaction to start AudioContext in modern browsers
    const initAudio = () => {
      this._init().then(() => {
        // Success! Remove the event listeners so we don't re-init
        document.removeEventListener('click', initAudio);
        document.removeEventListener('touchstart', initAudio);
        
        // Update UI to show we are listening
        this._onStateChange('idle'); 
      }).catch(err => {
        console.error('[Audio] Init failed:', err);
        alert("Microphone access is required. Please check your browser URL bar and allow microphone permissions, then reload.");
      });
    };
    
    // Listen for the very first click anywhere on the page
    document.addEventListener('click', initAudio);
    document.addEventListener('touchstart', initAudio);
    console.warn('[Audio] Waiting for user interaction (click anywhere) to request mic access...');
  }

  onMouth(cb) { this._mouthCallback = cb; }

  receiveAudio(arrayBuffer) {
    this._audioQueue.push(arrayBuffer);
    if (!this._isPlaying) this._playNext();
  }

  pauseMic() { this._micActive = false; }
  resumeMic() { this._micActive = true; }

  async _init() {
    if (this._micCtx) return;

    // Force exact sample rate for openwakeword compatibility (16000Hz)
    this._micCtx = new AudioContext({ sampleRate: 16000 });
    this._playbackCtx = new AudioContext(); 

    if (this._micCtx.state === 'suspended') await this._micCtx.resume();
    if (this._playbackCtx.state === 'suspended') await this._playbackCtx.resume();

    // 1. Explicitly prompt the browser for Microphone access
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

    // 2. Route the mic through a ScriptProcessor to capture raw bytes
    const source = this._micCtx.createMediaStreamSource(this._stream);
    
    // Create processor (bufferSize 4096, 1 input channel, 1 output channel)
    this._processor = this._micCtx.createScriptProcessor(4096, 1, 1);
    
    this._processor.onaudioprocess = (e) => {
      // Don't process/send mic data if the AI is currently speaking
      if (!this._micActive || this._isPlaying) return;

      const inputData = e.inputBuffer.getChannelData(0); // Float32Array [-1.0 to 1.0]
      const pcm16 = new Int16Array(inputData.length);
      
      // Convert Float32 to Int16 for Python's numpy format
      for (let i = 0; i < inputData.length; i++) {
        pcm16[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
      }
      
      // Send raw binary buffer to WebSocket
      this._sendBinary(pcm16.buffer);
    };

    // Connect nodes: Mic -> Processor -> Destination
    source.connect(this._processor);
    this._processor.connect(this._micCtx.destination);
    
    this._micActive = true;
    console.log('[Audio] Mic active. Raw 16kHz PCM streaming started successfully.');
  }

  // --- TTS Playback (Keep this exactly as you had it originally) ---
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