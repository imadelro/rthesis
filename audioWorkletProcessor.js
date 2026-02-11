// extension/audioWorkletProcessor.js
class ChunkProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this._buffer = [];
    // Default ~0.5s at 48kHz
    this.chunkSize = (options.processorOptions && options.processorOptions.chunkSize) || 24000;
    this.sampleRate = sampleRate;
  }

  process(inputs, outputs/*, parameters*/) {
    const input = inputs[0];
    const output = outputs[0];

    // Pass-through: copy input channel 0 to output channel 0 if present
    if (input && input[0] && output && output[0]) {
      const inCh0 = input[0];
      const outCh0 = output[0];
      // inCh0/outCh0 are Float32Array frames for each render quantum (128 frames)
      outCh0.set(inCh0);

      // Also accumulate for chunk emission to main thread
      // Clone because outCh0 will be reused next quantum
      this._buffer.push(new Float32Array(inCh0));
      let totalLen = 0;
      for (let i = 0; i < this._buffer.length; i++) totalLen += this._buffer[i].length;
      if (totalLen >= this.chunkSize) {
        const out = new Float32Array(totalLen);
        let offset = 0;
        for (let i = 0; i < this._buffer.length; i++) { out.set(this._buffer[i], offset); offset += this._buffer[i].length; }
        this.port.postMessage({ audioChunk: out, sampleRate: this.sampleRate });
        this._buffer = [];
      }
    } else if (output && output[0]) {
      // If no input, output silence
      output[0].fill(0);
    }

    return true;
  }
}

registerProcessor('chunk-processor', ChunkProcessor);
