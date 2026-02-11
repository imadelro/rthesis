from flask import Flask, request, jsonify
import numpy as np
import base64
from model_inference import predict_hate_speech

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def chunk():
    data = request.json
    audio_b64 = data.get('audio_base64')
    sr = data.get('sample_rate', 16000)
    if not audio_b64:
        return jsonify({'status': 'no audio'}), 400
    pcm_bytes = base64.b64decode(audio_b64)
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    try:
        transcript, label = predict_hate_speech(arr, sr)
        return jsonify({'status': 'ok', 'transcript': transcript, 'label': label})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)