from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import subprocess
import logging
import time
from logging import basicConfig
import shutil
import sys
import re
import io
from typing import Optional, Tuple
from faster_whisper import WhisperModel  # Use only Faster-Whisper

# --- Hate speech text classification (transcript-level) ---
HATE_MODEL_LOADED = False
HATE_CLF = None
HATE_TOKENIZER = None
HATE_ID2LABEL = None
HATE_MODEL_SOURCE = None  # resolved directory or model identifier
HATE_NUM_LABELS = None
HATE_MODEL_LAST_ERROR = None

def _write_current_hate_model_file():
    """Persist the currently active hate model source to a small text file for easy inspection."""
    try:
        models_dir = os.path.join(BASE_DIR, 'models')
        os.makedirs(models_dir, exist_ok=True)
        out_path = os.path.join(models_dir, 'current_hate_model.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(str(HATE_MODEL_SOURCE) if HATE_MODEL_SOURCE else 'None')
        logging.info('Wrote current hate model to %s', out_path)
    except Exception:
        logging.exception('Failed writing current_hate_model.txt')

def init_hate_model():
    """Lazy-load a local Transformers sequence classification model for hate speech.
    Auto-detects the provided local directory 'models/hate models/final_model' unless overridden by env var HATE_MODEL_DIR.
    Set HATE_MODEL_DIR to a HuggingFace Hub model name or absolute directory path containing config.json + model weights.
    """
    global HATE_MODEL_LOADED, HATE_CLF, HATE_TOKENIZER, HATE_ID2LABEL, HATE_MODEL_SOURCE, HATE_NUM_LABELS
    if HATE_MODEL_LOADED:
        return HATE_CLF, HATE_TOKENIZER, HATE_ID2LABEL
    model_dir_env = os.getenv('HATE_MODEL_DIR')
    if model_dir_env:
        candidate = model_dir_env
    else:
        # default local path with spaces in folder names
        candidate = os.path.join(BASE_DIR, 'models', 'hate models', 'final_model')
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig  # type: ignore
        resolved = candidate
        if os.path.isdir(candidate):
            resolved = os.path.abspath(candidate)
        t0 = time.perf_counter()
        logging.info('Loading hate speech model from %s', resolved)
        config = AutoConfig.from_pretrained(candidate)
        # Try to load model weights onto CPU explicitly to avoid 'meta' tensors
        try:
            import torch
            # Prefer explicit cpu device map to ensure weights are materialized
            HATE_CLF = AutoModelForSequenceClassification.from_pretrained(candidate, device_map='cpu', torch_dtype=torch.float32)
            HATE_CLF.to('cpu')
        except Exception:
            # Fallback to default load then move to CPU
            try:
                HATE_CLF = AutoModelForSequenceClassification.from_pretrained(candidate)
                try:
                    HATE_CLF.to('cpu')
                except Exception:
                    pass
            except Exception:
                # Final fallback: re-raise to be caught by outer handler
                raise
        HATE_TOKENIZER = AutoTokenizer.from_pretrained(candidate)
        try:
            HATE_CLF.eval()
        except Exception:
            pass
        logging.info('Hate model load time: %.2f ms', (time.perf_counter() - t0) * 1000)
        # id2label may be in config
        HATE_ID2LABEL = getattr(config, 'id2label', None)
        if not HATE_ID2LABEL and hasattr(config, 'label2id'):
            # invert label2id if present
            HATE_ID2LABEL = {v: k for k, v in config.label2id.items()}
        HATE_MODEL_SOURCE = resolved
        try:
            HATE_NUM_LABELS = int(getattr(config, 'num_labels', None)) if getattr(config, 'num_labels', None) is not None else None
        except Exception:
            HATE_NUM_LABELS = None
        HATE_MODEL_LOADED = True
        HATE_MODEL_LAST_ERROR = None
        logging.info('Hate model loaded: source=%s num_labels=%s id2label=%s', HATE_MODEL_SOURCE, HATE_NUM_LABELS, HATE_ID2LABEL)
        # Also print to the terminal/console so running `python server.py` shows the active model
        try:
            print(f"Hate model loaded: {HATE_MODEL_SOURCE} (num_labels={HATE_NUM_LABELS})")
        except Exception:
            pass
        _write_current_hate_model_file()
    except Exception as e:
        logging.exception('Failed to load hate speech model at %s', candidate)
        # Print error to terminal as well for quick visibility
        try:
            print(f"Failed to load hate model at {candidate}: {e}")
        except Exception:
            pass
        HATE_MODEL_LOADED = False
        HATE_MODEL_LAST_ERROR = str(e)
    return HATE_CLF, HATE_TOKENIZER, HATE_ID2LABEL


def classify_text_hate(transcript: str) -> Tuple[str, Optional[float]]:
    """Classify transcript and return (normalized_label, confidence_score or None).
    Normalized labels: Hate Speech | Offensive | Non-Hate Speech | Unknown
    """
    if not transcript or not transcript.strip():
        return ('Unknown', None)
    clf, tok, id2label = init_hate_model()
    if clf is None or tok is None:
        return ('Unknown', None)
    try:
        import torch  # type: ignore
        # Truncate overly long transcripts for classification to a reasonable token length
        inputs = tok(transcript, return_tensors='pt', truncation=True, max_length=512)
        t0 = time.perf_counter()  # start AFTER tokenization (model forward only)
        with torch.no_grad():
            # Ensure inputs are on same device as model parameters
            try:
                model_device = next(clf.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
            except Exception:
                pass
            outputs = clf(**inputs)
            logits = outputs.logits
            # Detect meta tensors (PyTorch >=2 may expose is_meta)
            if getattr(logits, 'is_meta', False):
                logging.warning('logits is a meta tensor; attempting to move model to CPU and re-run')
                try:
                    clf.to('cpu')
                    model_device = next(clf.parameters()).device
                    inputs = {k: v.to(model_device) for k, v in inputs.items()}
                    outputs = clf(**inputs)
                    logits = outputs.logits
                except Exception:
                    logging.exception('Failed to recover from meta tensor state')
                    raise
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            # Move to CPU and detach for safe indexing
            probs_cpu = probs.detach().cpu()
            pred_idx = int(torch.argmax(probs_cpu, dim=-1).item())
            score = float(probs_cpu[pred_idx].item())
        forward_ms = (time.perf_counter() - t0) * 1000
        logging.info('hate_classification_forward_ms=%.2f', forward_ms)
        raw_label = None
        if id2label and pred_idx in id2label:
            raw_label = str(id2label[pred_idx])
        # Fallback mapping if id2label absent
        if raw_label is None:
            raw_label = f'LABEL_{pred_idx}'
        # Special-case: binary models (num_labels == 2) with convention 0 = non-hate, 1 = hate
        # Avoid relying on textual label names that may be generic (e.g., 'LABEL_0').
        try:
            from math import isfinite  # noqa: F401 (guard import if optimized later)
            # Use global num_labels for decision; if absent but id2label has length 2, treat as binary.
            if (HATE_NUM_LABELS == 2) or (id2label and isinstance(id2label, dict) and len(id2label) == 2):
                if pred_idx == 1:
                    return ('Hate Speech', score)
                else:
                    return ('Non-Hate Speech', score)
        except Exception:
            # Fall through to heuristic mapping below if any issue
            pass
        raw_upper = raw_label.upper()
        # Normalize
        if 'HATE' in raw_upper:
            norm = 'Hate Speech'
        elif any(k in raw_upper for k in ['OFF', 'TOXIC', 'INSULT', 'ABUSE', 'HARASS']):
            norm = 'Offensive'
        elif 'NONE' in raw_upper or 'NEUTRAL' in raw_upper:
            norm = 'Non-Hate Speech'
        else:
            # Heuristic: treat single-label models without explicit names as Non-Hate
            norm = 'Non-Hate Speech'
        return (norm, score)
    except Exception:
        logging.exception('Hate speech classification failed')
        return ('Unknown', None)

# --- WebM remediation helpers ---
def ffmpeg_remux_webm(src_path: str, ffmpeg_path: str) -> str | None:
    """Attempt a lossless re-mux (stream copy) into a fresh WebM container to
    repair timing/index issues. Returns remuxed path or None on failure."""
    try:
        base, ext = os.path.splitext(src_path)
        if ext.lower() != '.webm':
            return None
        remux_path = base + '.remux.webm'
        cmd = [ffmpeg_path, '-y', '-fflags', '+genpts', '-i', src_path, '-c', 'copy', remux_path]
        conv = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if conv.returncode != 0:
            logging.warning('webm remux failed (code=%s): %s', conv.returncode, conv.stderr.decode('utf-8', errors='ignore')[:500])
            return None
        # Basic sanity: ensure remux file larger or equal
        try:
            if os.path.getsize(remux_path) < max(128, os.path.getsize(src_path) * 0.9):
                logging.warning('remux size suspicious; keeping original')
                return None
        except Exception:
            pass
        logging.info('webm remux successful: %s', remux_path)
        return remux_path
    except Exception:
        logging.exception('webm remux exception')
        return None

def ffmpeg_reencode_webm(src_path: str, ffmpeg_path: str) -> str | None:
    """Re-encode WebM audio to fresh Opus if remux fails (salvage partial files).
    Returns new path or None."""
    try:
        base, ext = os.path.splitext(src_path)
        if ext.lower() != '.webm':
            return None
        fixed_path = base + '.fixed.webm'
        cmd = [ffmpeg_path, '-y', '-i', src_path, '-vn', '-ar', '16000', '-ac', '1', '-c:a', 'libopus', '-b:a', '96k', fixed_path]
        conv = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if conv.returncode != 0:
            logging.warning('webm re-encode failed (code=%s): %s', conv.returncode, conv.stderr.decode('utf-8', errors='ignore')[:500])
            return None
        try:
            if os.path.getsize(fixed_path) < 128:
                logging.warning('re-encoded webm too small; discarding')
                return None
        except Exception:
            pass
        logging.info('webm re-encode successful: %s', fixed_path)
        return fixed_path
    except Exception:
        logging.exception('webm re-encode exception')
        return None

# Global Faster-Whisper model instance (loaded once)
FAST_MODEL = None

def init_fast_whisper():
    """Initialize Faster-Whisper model once (global)."""
    global FAST_MODEL
    if FAST_MODEL is not None:
        return FAST_MODEL
    model_name = os.getenv('FASTER_WHISPER_MODEL', 'small')
    compute_type = os.getenv('FASTER_WHISPER_COMPUTE', 'int8')  # int8 / int8_float16 / float16 / float32
    device = os.getenv('FASTER_WHISPER_DEVICE', 'cpu')
    logging.info('Initializing Faster-Whisper model=%s device=%s compute_type=%s', model_name, device, compute_type)
    FAST_MODEL = WhisperModel(model_name, device=device, compute_type=compute_type)
    return FAST_MODEL

app = Flask(__name__)
CORS(app)

# Configure logging to show INFO messages by default so we can see helper outputs
basicConfig(level=logging.INFO)

# Use absolute directories anchored at this file's location to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
TRANSCRIPTS_DIR = os.path.join(BASE_DIR, 'transcripts')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)



def ensure_ffmpeg_in_path():
    """Ensure ffmpeg is discoverable by transformers.
    If not on PATH, but models/ffmpeg.exe exists, prepend that directory.
    Returns path to ffmpeg if found, else None.
    """
    found = shutil.which('ffmpeg')
    if found:
        return found
    candidate = os.path.join(BASE_DIR, 'models', 'ffmpeg.exe')
    if os.path.isfile(candidate):
        models_dir = os.path.dirname(candidate)
        if models_dir not in os.environ.get('PATH', ''):
            os.environ['PATH'] = models_dir + os.pathsep + os.environ.get('PATH', '')
        logging.info('Added local ffmpeg to PATH: %s', candidate)
        return candidate
    logging.warning('ffmpeg not found; HF pipeline may fail. Place ffmpeg.exe in server/models/')
    return None
# Initialize Faster-Whisper once at import
try:
    init_fast_whisper()
    logging.info('Global Faster-Whisper model initialized.')
except Exception:
    logging.exception('Failed to initialize global Faster-Whisper model')



def next_index_for_id(video_id: str) -> int:
    """Return the next sequential integer index for the given video_id based on
    existing files in uploads/ and transcripts/.

    Looks for files named like '<video_id>-<n>.<ext>' and returns max(n)+1, or 1 if none.
    """
    max_idx = 0
    pattern = re.compile(r'^' + re.escape(video_id) + r'-(\d+)\b')
    try:
        for directory in (UPLOAD_DIR, TRANSCRIPTS_DIR):
            if not os.path.isdir(directory):
                continue
            for name in os.listdir(directory):
                m = pattern.match(name)
                if m:
                    try:
                        idx = int(m.group(1))
                        if idx > max_idx:
                            max_idx = idx
                    except Exception:
                        continue
    except Exception:
        # On any error, fall back to conservative next index = 1
        pass
    return max_idx + 1 if max_idx >= 0 else 1


@app.route('/status', methods=['GET'])
def status():
    """Return basic environment and autodetect info to help debugging which
    transcription path will be used.
    """
    # Ensure hate model is initialized so status reflects the true active model
    try:
        init_hate_model()
    except Exception:
        logging.exception('init_hate_model failed during /status')
    return {
        # Hate model reporting
        'hate_model_loaded': HATE_MODEL_LOADED,
        'hate_model_source': HATE_MODEL_SOURCE,
        'hate_num_labels': HATE_NUM_LABELS,
        'hate_id2label': HATE_ID2LABEL,
        'hate_model_error': HATE_MODEL_LAST_ERROR,
        # Faster-Whisper reporting
        'faster_whisper_model': os.getenv('FASTER_WHISPER_MODEL', 'small'),
        'faster_whisper_device': os.getenv('FASTER_WHISPER_DEVICE', 'cpu'),
        'faster_whisper_compute': os.getenv('FASTER_WHISPER_COMPUTE', 'int8'),
    }




@app.route('/list_uploads', methods=['GET'])
def list_uploads():
    """Return a JSON list of files currently in the uploads directory with basic metadata.
    This helps confirm where files are being saved and their sizes/timestamps.
    """
    try:
        files = []
        for name in os.listdir(UPLOAD_DIR):
            path = os.path.join(UPLOAD_DIR, name)
            try:
                stat = os.stat(path)
                entry = {
                    'name': name,
                    'size': stat.st_size,
                    'mtime': int(stat.st_mtime),
                    'path': os.path.abspath(path)
                }
                # If a transcript exists for this upload, include its path
                base, _ = os.path.splitext(name)
                transcript_candidate = os.path.join(TRANSCRIPTS_DIR, base + '.txt')
                if os.path.isfile(transcript_candidate):
                    entry['transcript'] = os.path.abspath(transcript_candidate)
                files.append(entry)
            except Exception:
                files.append({'name': name, 'error': 'stat-failed'})
        # sort newest first
        files.sort(key=lambda x: x.get('mtime', 0), reverse=True)
        return jsonify({'uploads': files})
    except FileNotFoundError:
        return jsonify({'uploads': [], 'error': 'upload-dir-missing'}), 404
    except Exception as e:
        logging.exception('list_uploads failed')
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Supports two modes:
    # 1) Streaming raw audio in request body (preferred; stays in-memory)
    # 2) Legacy file upload via form field 'file' (fallback)

    model = init_fast_whisper()
    beam_size = int(os.getenv('FASTER_WHISPER_BEAM', '1'))

    # Preferred streaming path: no files, raw audio in body
    if 'file' not in request.files:
        content_type = (request.headers.get('Content-Type') or '').lower()
        t_recv = time.time()  # ms epoch optional
        p_recv = time.perf_counter()
        stream_id = request.headers.get('X-Stream-Id') or request.args.get('streamId') or request.form.get('streamId') or 'stream'
        # Buffer request body into memory
        buf = io.BytesIO()
        try:
            p_stream_read_start = time.perf_counter()
            while True:
                chunk = request.stream.read(65536)
                if not chunk:
                    break
                buf.write(chunk)
            buf.seek(0)
            p_stream_read_end = time.perf_counter()
        except Exception:
            logging.exception('Failed reading streaming body')
            return jsonify({'error': 'stream-read-failed'}), 400

        # Decode to numpy float32 mono 16k in-memory
        audio_arr = None
        duration_sec = None
        try:
            p_decode_start = time.perf_counter()
            if 'audio/wav' in content_type or (request.headers.get('X-Audio-Format','').lower() == 'wav'):
                import wave, numpy as np
                with wave.open(buf, 'rb') as wf:
                    sr = wf.getframerate()
                    ch = wf.getnchannels()
                    n = wf.getnframes()
                    frames = wf.readframes(n)
                samples = np.frombuffer(frames, dtype=np.int16)
                if ch and ch > 1:
                    samples = samples.reshape(-1, ch).mean(axis=1).astype(np.int16)
                # simple linear resample to 16k if needed
                if sr != 16000 and sr > 0:
                    import numpy as np
                    x = np.arange(len(samples))
                    new_len = max(1, int(len(samples) * 16000 / sr))
                    new_x = np.linspace(0, len(samples) - 1, new_len)
                    samples = np.interp(new_x, x, samples).astype(np.int16)
                audio_arr = samples.astype('float32') / 32768.0
                duration_sec = len(audio_arr) / 16000.0
            else:
                # Assume PCM S16LE stream with headers specifying rate/channels
                import numpy as np
                sr = int(request.headers.get('X-Sample-Rate', '16000'))
                ch = int(request.headers.get('X-Channels', '1'))
                buf.seek(0)
                raw = buf.read()
                samples = np.frombuffer(raw, dtype=np.int16)
                if ch and ch > 1:
                    samples = samples.reshape(-1, ch).mean(axis=1).astype(np.int16)
                if sr != 16000 and sr > 0:
                    x = np.arange(len(samples))
                    new_len = max(1, int(len(samples) * 16000 / sr))
                    new_x = np.linspace(0, len(samples) - 1, new_len)
                    samples = np.interp(new_x, x, samples).astype(np.int16)
                audio_arr = samples.astype('float32') / 32768.0
                duration_sec = len(audio_arr) / 16000.0
            p_decode_end = time.perf_counter()
        except Exception:
            logging.exception('Failed to decode streaming audio')
            return jsonify({'error': 'decode-failed', 'hint': 'Send audio/wav or PCM S16LE with X-Sample-Rate and X-Channels headers'}), 400

        # End-to-end timer starts when upload (stream read) finishes
        e2e_start = time.perf_counter()
        try:
            logging.info('Transcribing (stream) with Faster-Whisper beam_size=%d', beam_size)
            t0 = time.perf_counter()
            segments, info = model.transcribe(audio_arr, beam_size=beam_size, language=None)
            t_infer_ms = int((time.perf_counter() - t0) * 1000)
            logging.info('transcription_inference_ms=%d (stream)', t_infer_ms)
            transcript_text = ' '.join(seg.text.strip() for seg in segments)
            # Persist transcript only (no audio written)
            base = f"{stream_id}-{int(time.time())}"
            p_write_start = time.perf_counter()
            tpath = os.path.join(TRANSCRIPTS_DIR, base + '.txt')
            with open(tpath, 'w', encoding='utf-8') as tf:
                tf.write(transcript_text)
            p_write_end = time.perf_counter()
            logging.info('Saved transcript to %s', tpath)
            c0 = time.perf_counter()
            hate_label, hate_score = classify_text_hate(transcript_text)
            t_class_ms = int((time.perf_counter() - c0) * 1000)
            e2e_ms = int((time.perf_counter() - e2e_start) * 1000)
            logging.info('end_to_end_latency_ms=%d (stream) audio_duration_s=%.2f classification_wrapper_ms=%d', e2e_ms, (duration_sec if duration_sec else (getattr(info, 'duration', 0) or 0.0)), t_class_ms)
            resp = {
                'method': 'faster_whisper_stream',
                'transcript': transcript_text,
                'duration': duration_sec if duration_sec is not None else getattr(info, 'duration', None),
                'transcript_path': os.path.abspath(tpath),
                'hateLabel': hate_label,
                'hateScore': hate_score,
                'hateModel': HATE_MODEL_SOURCE,
                'inferenceMs': t_infer_ms,
                'classificationMs': t_class_ms,
                'endToEndMs': e2e_ms,
                'timingsMs': {
                    'requestStreamReadMs': int((p_stream_read_end - p_stream_read_start) * 1000),
                    'decodeAudioMs': int((p_decode_end - p_decode_start) * 1000),
                    'writeTranscriptMs': int((p_write_end - p_write_start) * 1000),
                },
                'timestampsMs': {
                    'tRecv': int(t_recv * 1000),
                }
            }
            return jsonify(resp)
        except Exception as e:
            logging.exception('Faster-Whisper streaming transcription failed')
            return jsonify({'error': 'faster-whisper failed', 'detail': str(e)}), 500

    # --- Legacy file upload path (kept for backward-compat) ---
    f = request.files['file']

    # Dynamic hate model switching removed; always use initially loaded model.
    # Determine naming based on provided videoId and whether this is the final upload for that video
    form_video_id = request.form.get('videoId') or 'anon'
    form_final = request.form.get('final') or '0'
    # Extension from incoming filename (default to .webm for safety)
    _, incoming_ext = os.path.splitext(f.filename)
    if not incoming_ext:
        incoming_ext = '.webm'
    # For final uploads, save as '<videoId>-<n>.<ext>' where n is sequential per videoId
    # For segments, keep them as 'segment-<videoId>-<ts>.<ext>' so we can merge later
    sequence_index = None
    if form_final == '1':
        idx = next_index_for_id(form_video_id)
        sequence_index = idx
        base_no_ext = f"{form_video_id}-{idx}"
    else:
        base_no_ext = f"segment-{form_video_id}-{int(time.time() * 1000)}"
    filename = os.path.join(UPLOAD_DIR, base_no_ext + incoming_ext)
    # Ensure destination directory exists (defensive against any runtime CWD changes)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    p_save_start = time.perf_counter()
    f.save(filename)
    p_save_end = time.perf_counter()
    # Start end-to-end timer immediately after upload save completes
    e2e_start = time.perf_counter()
    # If the client provided a videoId and indicated this is the final
    # fragment for that video, attempt to assemble previously uploaded
    # segment files for the same video into a single merged file and use
    # that for transcription. This helps when the final assembled blob
    # sent by the browser is short but the per-segment uploads together
    # represent the full audio for the video.
    # Note: we already extracted form_video_id/form_final above for naming.
    # Keep a None form_video_id when missing to preserve existing flow.
    form_video_id = (request.form.get('videoId') or None)
    merged_candidate = None
    if form_video_id and form_final == '1':
        logging.info('Final upload for videoId=%s received; looking for segments to merge', form_video_id)
        try:
            # find segment files that include the video id in their name
            segs = []
            for nm in os.listdir(UPLOAD_DIR):
                if form_video_id in nm and (nm.startswith('segment-') or nm.startswith('recording-')):
                    segs.append(os.path.join(UPLOAD_DIR, nm))
            segs.sort(key=lambda p: os.path.getmtime(p))
            if segs:
                # create a concat list for ffmpeg
                list_path = os.path.join(UPLOAD_DIR, f'concat-{form_video_id}-{int(time.time())}.txt')
                with open(list_path, 'w', encoding='utf-8') as lf:
                    for s in segs:
                        # ffmpeg concat file expects lines like: file '/absolute/path'
                        lf.write("file '" + os.path.abspath(s).replace('\\\\', '/') + "'\n")

                # prefer ffmpeg on PATH or local models/ffmpeg.exe
                ffmpeg_path = shutil.which('ffmpeg')
                if not ffmpeg_path:
                    local_ffmpeg = os.path.join(BASE_DIR, 'models', 'ffmpeg.exe')
                    if os.path.isfile(local_ffmpeg):
                        ffmpeg_path = os.path.abspath(local_ffmpeg)

                if ffmpeg_path:
                    # Save merged audio using the same base name as the uploaded final (e.g., '<videoId>-<n>.wav')
                    base_name_no_ext = os.path.splitext(os.path.basename(filename))[0]
                    merged_candidate = os.path.join(UPLOAD_DIR, base_name_no_ext + '.wav')
                    logging.info('Attempting to merge %d segments into %s using ffmpeg at %s', len(segs), merged_candidate, ffmpeg_path)
                    conv = subprocess.run([ffmpeg_path, '-y', '-f', 'concat', '-safe', '0', '-i', list_path,
                                           '-vn', '-ar', '16000', '-ac', '1', '-acodec', 'pcm_s16le', '-f', 'wav', merged_candidate],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                    if conv.returncode != 0:
                        ff_err = conv.stderr.decode('utf-8', errors='ignore')
                        logging.error('ffmpeg merge failed: %s', ff_err)
                        # Save stderr for inspection and leave merged_candidate as None
                        try:
                            errlog = os.path.join(UPLOAD_DIR, f'concat-{form_video_id}.ffmpeg.log')
                            with open(errlog, 'w', encoding='utf-8') as ef:
                                ef.write(ff_err)
                            logging.info('Saved ffmpeg concat stderr to %s', errlog)
                        except Exception:
                            logging.exception('Failed to write ffmpeg concat stderr log')
                        merged_candidate = None
                    else:
                        logging.info('ffmpeg merge successful: %s', merged_candidate)
                else:
                    logging.info('ffmpeg not available; cannot merge segments for videoId=%s', form_video_id)
        except Exception:
            logging.exception('Failed while attempting to merge segments')
    # If merging succeeded, prefer the merged file as the input
    if merged_candidate and os.path.isfile(merged_candidate):
        filename = merged_candidate
    try:
        size = os.path.getsize(filename)
        logging.info('Saved upload %s (%d bytes)', filename, size)
    except Exception:
        logging.info('Saved upload %s', filename)
    original_filename = filename
    # If uploaded file is a container (webm/ogg/m4a), try converting to WAV with ffmpeg
    name, ext = os.path.splitext(filename)
    ext = ext.lower()
    converted = None
    temp_generated_paths = []  # track intermediate files to clean up later
    p_convert_total_start = time.perf_counter()
    if ext in ('.webm', '.ogg', '.m4a', '.mp4'):
        # prefer an ffmpeg on PATH, but also accept a local copy in server/models
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            local_ffmpeg = os.path.join(BASE_DIR, 'models', 'ffmpeg.exe')
            if os.path.isfile(local_ffmpeg):
                ffmpeg_path = os.path.abspath(local_ffmpeg)
                logging.info('Using local ffmpeg at %s', ffmpeg_path)
        if ffmpeg_path:
            # If it's a webm, attempt a lossless remux first to repair timestamps
            if ext == '.webm':
                remuxed = ffmpeg_remux_webm(filename, ffmpeg_path)
                if remuxed:
                    temp_generated_paths.append(remuxed)
                    filename = remuxed
                else:
                    # If remux failed, try re-encode salvage
                    salvage = ffmpeg_reencode_webm(filename, ffmpeg_path)
                    if salvage:
                        temp_generated_paths.append(salvage)
                        filename = salvage
            converted = f"{name}.wav"
            try:
                logging.info('Converting %s -> %s using ffmpeg', filename, converted)
                p_ffmpeg_start = time.perf_counter()
                conv = subprocess.run([ffmpeg_path, '-y', '-i', filename, '-vn', '-ar', '16000', '-ac', '1', '-acodec', 'pcm_s16le', '-f', 'wav', converted],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                p_ffmpeg_end = time.perf_counter()
                if conv.returncode != 0:
                    err_txt = conv.stderr.decode('utf-8', errors='ignore')
                    logging.error('ffmpeg conversion failed: %s', err_txt)
                    # Save stderr for inspection
                    try:
                        base = os.path.splitext(os.path.basename(filename))[0]
                        errlog = os.path.join(UPLOAD_DIR, f'{base}.ffmpeg.log')
                        with open(errlog, 'w', encoding='utf-8') as ef:
                            ef.write(err_txt)
                        logging.info('Saved ffmpeg stderr to %s', errlog)
                    except Exception:
                        logging.exception('Failed to write ffmpeg conversion stderr log')
                    converted = None
                else:
                    logging.info('ffmpeg conversion successful: %s', converted)
                    # Validate WAV container quickly to avoid "malformed soundfile" surprises downstream
                    try:
                        import wave
                        with wave.open(converted, 'rb') as wf:
                            _sr = wf.getframerate()
                            _n = wf.getnframes()
                            _ch = wf.getnchannels()
                            if _n == 0:
                                raise ValueError('empty wav (0 frames)')
                            if _ch not in (1, 2):
                                raise ValueError(f'unexpected channels: {_ch}')
                        filename = converted
                        # Clean up original container and any intermediates — keep only audio WAV
                        try:
                            # Remove the initially uploaded container (e.g., .webm/.mp4)
                            if os.path.isfile(original_filename) and os.path.abspath(original_filename) != os.path.abspath(converted):
                                os.remove(original_filename)
                        except Exception:
                            logging.exception('Failed to delete original upload %s', original_filename)
                        # Remove any temp remux/salvage files
                        for p in temp_generated_paths:
                            try:
                                if os.path.isfile(p) and os.path.abspath(p) != os.path.abspath(converted):
                                    os.remove(p)
                            except Exception:
                                logging.exception('Failed to delete temp file %s', p)
                    except Exception as wav_e:
                        logging.error('Converted WAV failed validation (%s); falling back to original input %s', wav_e, original_filename)
                        converted = None
            except Exception:
                logging.exception('ffmpeg conversion exception')
                converted = None
        else:
            logging.info('ffmpeg not found on PATH; skipping conversion for %s', filename)
    p_convert_total_end = time.perf_counter()
    # Transcribe using Faster-Whisper only
    try:
        model = init_fast_whisper()
        beam_size = int(os.getenv('FASTER_WHISPER_BEAM', '1'))
        logging.info('Transcribing with Faster-Whisper beam_size=%d language=%s', beam_size, 'auto')
        t0 = time.perf_counter()
        segments, info = model.transcribe(filename, beam_size=beam_size, language=None)
        t_infer_ms = int((time.perf_counter() - t0) * 1000)
        logging.info('transcription_inference_ms=%d (file)', t_infer_ms)
        transcript_text = ' '.join(seg.text.strip() for seg in segments)
        p_write_start = time.perf_counter()
        base = os.path.splitext(os.path.basename(original_filename))[0]
        tpath = os.path.join(TRANSCRIPTS_DIR, base + '.txt')
        with open(tpath, 'w', encoding='utf-8') as tf:
            tf.write(transcript_text)
        p_write_end = time.perf_counter()
        logging.info('Saved transcript to %s', tpath)
        c0 = time.perf_counter()
        hate_label, hate_score = classify_text_hate(transcript_text)
        t_class_ms = int((time.perf_counter() - c0) * 1000)
        e2e_ms = int((time.perf_counter() - e2e_start) * 1000)
        # Duration may come from info if available
        audio_duration = getattr(info, 'duration', None)
        logging.info('end_to_end_latency_ms=%d (file) audio_duration_s=%s classification_wrapper_ms=%d', e2e_ms, ('%.2f' % audio_duration) if audio_duration else 'unknown', t_class_ms)
        resp = {
            'method': 'faster_whisper',
            'transcript': transcript_text,
            'duration': getattr(info, 'duration', None),
            'transcript_path': os.path.abspath(tpath),
            'hateLabel': hate_label,
            'hateScore': hate_score,
            'hateModel': HATE_MODEL_SOURCE,
            'inferenceMs': t_infer_ms,
            'classificationMs': t_class_ms,
            'endToEndMs': e2e_ms,
            'timingsMs': {
                'saveUploadMs': int((p_save_end - p_save_start) * 1000),
                'formatConversionTotalMs': int((p_convert_total_end - p_convert_total_start) * 1000),
                'ffmpegConvertMs': int((p_ffmpeg_end - p_ffmpeg_start) * 1000) if 'p_ffmpeg_end' in locals() and 'p_ffmpeg_start' in locals() else None,
                'writeTranscriptMs': int((p_write_end - p_write_start) * 1000),
            },
        }
        if sequence_index is not None:
            resp['sequenceIndex'] = sequence_index
        return jsonify(resp)
    except Exception as e:
        logging.exception('Faster-Whisper transcription failed')
        return jsonify({'error': 'faster-whisper failed', 'detail': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
