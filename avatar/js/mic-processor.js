class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 4096;
    this.buffer = new Float32Array(this.bufferSize);
    this.bytesWritten = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0];

    for (let i = 0; i < channelData.length; i++) {
      this.buffer[this.bytesWritten++] = channelData[i];
      if (this.bytesWritten >= this.bufferSize) {
        this.flush();
      }
    }
    return true; 
  }

  flush() {
    if (this.bytesWritten === 0) return;
    const pcm16 = new Int16Array(this.bytesWritten);
    for (let i = 0; i < this.bytesWritten; i++) {
      pcm16[i] = Math.max(-32768, Math.min(32767, this.buffer[i] * 32768));
    }
    this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    this.bytesWritten = 0;
  }
}

registerProcessor('mic-processor', MicProcessor);