# nb_whisper

Windows dual-source live transcription tool.

## Features

- faster-whisper speech recognition
- Microphone capture with sounddevice
- Windows system audio loopback capture with soundcard
- HTML dashboard served at `http://127.0.0.1:8765`
- Optional Ollama/OpenAI-compatible Korean AI context generation
- Lyric Story Board context fields

## Quick Start

```bat
voice_game.bat small --device cpu --compute-type int8
```

Open:

```text
http://127.0.0.1:8765
```

## GPU Check

```bat
check_whisper_gpu.bat
```

For GPU inference with faster-whisper/CTranslate2, CUDA 12 cuBLAS and cuDNN 9 DLLs must be available in PATH:

- `cublas64_12.dll`
- `cudnn64_9.dll`

If they are missing, run with CPU:

```bat
voice_game.bat small --device cpu --compute-type int8
```

## Main Files

- `voice_dual_server.py`: current dual-source STT server
- `voice_game.html`: unified dashboard
- `voice_game.bat`: launcher
- `check_whisper_gpu.bat`: GPU dependency checker
- `voice_game_server.py`, `korean_mic_stt.py`: earlier voice context implementation
