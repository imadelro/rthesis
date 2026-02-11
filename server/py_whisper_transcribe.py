#!/usr/bin/env python3
"""
py_whisper_transcribe.py

Small helper that uses the Python OpenAI Whisper package to transcribe an audio file.
Usage: python py_whisper_transcribe.py /path/to/audio.wav

Environment:
  WHISPER_PY_MODEL - model name to load (default: small)

This script prints the transcript to stdout on success. On failure it prints
errors to stderr and exits with non-zero code.
"""
import os
import sys
import json
import traceback

def main():
    if len(sys.argv) < 2:
        print('usage: py_whisper_transcribe.py <audio-file>', file=sys.stderr)
        sys.exit(2)

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f'file not found: {path}', file=sys.stderr)
        sys.exit(3)

    # If there's a local ffmpeg binary bundled in server/models, make sure
    # it's on PATH so the HF transformers pipeline can invoke it when
    # loading audio files. This avoids "ffmpeg was not found" errors.
    try:
        # The ffmpeg binary is bundled in server/models/ffmpeg.exe. When this
        # helper runs we need to ensure that directory is on PATH so the HF
        # transformers pipeline (which spawns `ffmpeg`) can find it.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_models = os.path.join(script_dir, 'models')
        local_ffmpeg = os.path.join(local_models, 'ffmpeg.exe')
        # Debug: print whether the models dir and ffmpeg binary actually exist
        try:
            print(f'Local models dir: {local_models}', file=sys.stderr)
            print(f'Exists (dir): {os.path.isdir(local_models)}', file=sys.stderr)
            if os.path.isdir(local_models):
                try:
                    print('models dir listing: ' + ', '.join(os.listdir(local_models)), file=sys.stderr)
                except Exception:
                    print('models dir listing: <failed to list>', file=sys.stderr)
            print(f'Checking ffmpeg path: {local_ffmpeg} -> exists: {os.path.isfile(local_ffmpeg)}', file=sys.stderr)
        except Exception:
            # ignore failures in debug printing
            pass

        if os.path.isfile(local_ffmpeg):
            # Prepend to PATH for this process only
            os.environ['PATH'] = local_models + os.pathsep + os.environ.get('PATH', '')
            print(f'Added local ffmpeg to PATH: {local_ffmpeg}', file=sys.stderr)
    except Exception:
        # Non-fatal; pipeline will try to use ffmpeg on PATH normally
        pass

    model_name = os.getenv('WHISPER_PY_MODEL', 'small')
    # normalize model_name once

    # Only use Hugging Face transformers pipeline (openai/whisper-large-v3 or
    # compatible) for transcription. The previous fallback to the local
    # `whisper` python package has been removed per configuration.
    use_hf = model_name.startswith('openai/') or os.getenv('USE_HF_WHISPER') == '1'
    if not use_hf:
        print('This helper is configured to use Hugging Face openai/* models only.\n'
              'Set WHISPER_PY_MODEL to an openai/ model (e.g. openai/whisper-large-v3)\n'
              'or set USE_HF_WHISPER=1 to enable HF path.', file=sys.stderr)
        sys.exit(10)

    try:
        # local import to avoid requiring transformers when not used
        from transformers import pipeline
        import torch
    except Exception as e:
        print('failed to import transformers/torch for HF whisper: ' + str(e), file=sys.stderr)
        sys.exit(4)
    # If imports succeeded, run the HF transformers pipeline for transcription
    try:
        device = 0 if torch.cuda.is_available() else -1
        # determine task/language from env so users can control behavior
        task = os.getenv('WHISPER_PY_TASK', 'transcribe')
        lang_env = os.getenv('WHISPER_PY_LANG')
        dbg_lang = lang_env if lang_env else 'auto'
        print(f'py_whisper_transcribe: using HF transformers pipeline model={model_name} device={device} task={task} language={dbg_lang}', file=sys.stderr)
        # print versions to help debug crashes
        try:
            import transformers
            print(f'transformers={transformers.__version__} torch={torch.__version__}', file=sys.stderr)
        except Exception:
            pass

        asr = pipeline('automatic-speech-recognition', model=model_name, device=device)
        # build kwargs for the pipeline call; chunk_length_s helps with long audio
        call_kwargs = {'chunk_length_s': 30}
        # If the user explicitly set a WHISPER_PY_LANG, pass it to HF pipeline
        if lang_env:
            call_kwargs['language'] = lang_env

        result = asr(path, **call_kwargs)
        # transformers pipeline returns a dict with 'text'
        text = result.get('text') if isinstance(result, dict) else str(result)
        print(text)
        sys.exit(0)
    except Exception as e:
        print('HF transformers transcription failed: ' + str(e), file=sys.stderr)
        # include traceback for richer debugging information
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(6)

if __name__ == '__main__':
    main()
