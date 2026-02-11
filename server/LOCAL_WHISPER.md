Local whisper.cpp setup
=======================

This file explains how to set up `whisper.cpp` for local transcription on the server.

1) Download whisper.cpp binary
- Visit the releases page: https://github.com/ggerganov/whisper.cpp/releases
- Download a prebuilt binary for your platform (Windows: `main.exe` or similar). Place it somewhere like `C:\tools\whisper.cpp\main.exe`.

2) Download a ggml model
- From the same project or model mirrors, download a model such as `ggml-small.bin` or `ggml-medium.bin`.
- Place it under `server/models/`.

3) Configure environment variables (PowerShell example)

```powershell
# point to your whisper.cpp binary
$env:WHISPER_CPP_PATH = 'C:\tools\whisper.cpp\main.exe'
# optional: point to the model file (default: server/models/ggml-small.bin)
$env:WHISPER_MODEL_PATH = 'C:\path\to\models\ggml-small.bin'
# enable local whisper usage
$env:USE_LOCAL_WHISPER = '1'

# Run the server in the same shell so env vars are available
python server.py
```

4) Test transcription
- Use the extension to record audio and stop. The server will execute the whisper.cpp binary and return the stdout as the transcript.

Notes and troubleshooting
- If you see errors like "binary not found", double-check `WHISPER_CPP_PATH` and ensure the binary is executable.
- For Taglish, prefer `ggml-medium.bin` for better accuracy, but be aware it is slower.
- If you want to avoid using whisper.cpp, unset `USE_LOCAL_WHISPER` and set `OPENAI_API_KEY` instead to use the cloud Whisper.
