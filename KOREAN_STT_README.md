# Korean Mic To Text

Run only this file:

```cmd
korean_mic_to_text.bat
```

It automatically creates a virtual environment, installs packages, runs a quick microphone/runtime test, downloads the Whisper model on first run, and prints Korean microphone speech as text in the console.

It captures both sources by default:

- microphone
- system sound playing through your speakers/headset

Ollama DeepSeek usage:

- Whisper converts microphone audio to Korean text first.
- Ollama `deepseek-v3.1:671b-cloud` explains the recognized Korean text afterward.
- Default AI provider is Ollama at `http://localhost:11434`.
- Default AI mode is commentary/explanation.

## Custom Examples

Faster but lower accuracy:

```cmd
--model base
```

Better accuracy but slower:

```cmd
--model medium
```

Lower latency:

```cmd
--model small --chunk-sec 3
```

Use a smaller/faster model by passing it as the first argument:

```cmd
korean_mic_to_text.bat base
```

Lower silence threshold if it keeps printing `silence`:

```cmd
korean_mic_to_text.bat small --min-rms 0.0008
```

If it keeps printing `no speech detected`, try:

```cmd
korean_mic_to_text.bat small --no-speech-threshold 0.05 --gain 2
```

Pick a specific microphone:

```cmd
korean_mic_to_text.bat small --input-device 3
```

Capture only microphone:

```cmd
korean_mic_to_text.bat small --source mic
```

Capture only system sound:

```cmd
korean_mic_to_text.bat small --source system
```

Use DeepSeek correction plus explanation:

```cmd
korean_mic_to_text.bat small --ai-mode both
```

Use commentary/explanation only:

```cmd
korean_mic_to_text.bat small --ai-mode explain
```

If Ollama commentary times out, use a smaller/faster Ollama model or increase timeout:

```cmd
korean_mic_to_text.bat base --ai-model qwen2.5:7b --ai-timeout 120
```

Use simple correction only:

```cmd
korean_mic_to_text.bat small --ai-mode correct
```

Experimental Ollama STT mode:

```cmd
korean_mic_to_text.bat --stt-engine ollama --ollama-stt-model karanchopda333/whisper --source mic
```

This requires Ollama server access at `http://localhost:11434`. Audio support for Ollama STT models may not work with every model.

## Notes

- This is a development/test version.
- Audio is processed in RAM chunks.
- No recording file is created.
- Default model is `small`.
- The first model download requires internet.
