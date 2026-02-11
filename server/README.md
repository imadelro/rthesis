Quick test transcription server

This mock server accepts an uploaded audio file at POST /transcribe and returns a fake transcript for development/testing.

Setup (Windows / PowerShell):

1. Create a virtual environment and activate it (optional but recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the server:

```powershell
python server.py
```

The server will listen on http://127.0.0.1:5000 and accept uploads at /transcribe.

This server uses Hugging Face Transformers' automatic-speech-recognition pipeline by default. You can optionally enable Faster-Whisper for speed.

Performance & Optimization
--------------------------
The default Hugging Face `openai/whisper-large-v3` (or other large models) can be slow, especially on CPU. This project now supports several speed-ups:

1. Persistent model load (HF pipeline): Set an environment variable to choose a smaller model and avoid re-loading per request.
	Example (PowerShell):
	```powershell
	$env:WHISPER_PY_MODEL = 'openai/whisper-small'
	$env:USE_PY_WHISPER = '1'
	```
	Smaller variants (`openai/whisper-tiny`, `openai/whisper-base`, `openai/whisper-small`) are much faster. Use `openai/whisper-small.en` if you only need English.

2. Faster-Whisper (CTranslate2 backend): Enables quantized inference with large speed-ups on CPU and GPU.
	- Install dependency (already in requirements.txt): `pip install faster-whisper`
	- Enable via env:
	  ```powershell
	  $env:USE_FASTER_WHISPER = '1'
	  $env:FASTER_WHISPER_MODEL = 'small'   # tiny / base / small / medium / large-v3
	  $env:FASTER_WHISPER_COMPUTE = 'int8'  # int8 (fast) | int8_float16 | float16 | float32
	  $env:FASTER_WHISPER_DEVICE = 'auto'   # auto / cpu / cuda
	  ```
	- For best English-only speed on CPU use: model `tiny.en` and compute `int8`.

3. Chunk length: Adjust how audio is split for HF pipeline:
	```powershell
	$env:WHISPER_CHUNK_LEN = '20'  # default 30; lowering may reduce peak memory
	```

4. GPU acceleration: If you have an NVIDIA GPU and CUDA-enabled PyTorch installed, the server auto-selects GPU (`device=0`). Confirm with:
	```powershell
	python -c "import torch; print(torch.cuda.is_available())"
	```

5. Model selection strategy:
	- Short TikTok clips (< 60s): `tiny` or `base` often sufficient.
	- Mixed Tagalog/English: prefer `small` for better multilingual accuracy.
	- Use `FASTER_WHISPER_COMPUTE=int8_float16` on GPU for balanced speed/accuracy.

6. Avoid per-request reload: The server now caches the HF pipeline and faster-whisper model globally. Ensure you use the new env flags (`USE_FASTER_WHISPER` or `USE_PY_WHISPER`) rather than invoking the legacy helper script.

7. Preference order:
	- If `USE_FASTER_WHISPER=1` and dependency present: Faster-Whisper (CTranslate2 backend).
	- Else if `USE_PY_WHISPER=1` or `WHISPER_PY_MODEL` set: Hugging Face pipeline.

Example optimized local setup (English-only, CPU):
```powershell
$env:USE_FASTER_WHISPER = '1'
$env:FASTER_WHISPER_MODEL = 'tiny.en'
$env:FASTER_WHISPER_COMPUTE = 'int8'
python server.py
```

Example multilingual GPU (if CUDA available):
```powershell
$env:USE_FASTER_WHISPER = '1'
$env:FASTER_WHISPER_MODEL = 'small'
$env:FASTER_WHISPER_COMPUTE = 'int8_float16'
python server.py
```

Troubleshooting speed:
| Symptom | Suggestion |
|---------|------------|
| High CPU usage | Switch to `USE_FASTER_WHISPER=1` with `int8` quantization |
| Slow first request | Model cold load; subsequent requests are faster (warm cache) |
| Memory errors | Use smaller model (`tiny`/`base`) or lower `WHISPER_CHUNK_LEN` |
| Poor Tagalog accuracy | Increase model size to `small` or disable English-only variant |

The popup no longer displays transcripts; they are still saved to `server/transcripts/` for offline review.