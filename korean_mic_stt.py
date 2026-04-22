from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import re
import sys
import threading
import time
import warnings
import wave
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", message=".*data discontinuity in recording.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"soundcard\..*")


@dataclass
class SttConfig:
    model_size: str = "small"
    stt_engine: str = "whisper"
    ollama_stt_model: str = "karanchopda333/whisper"
    ollama_base_url: str = "http://localhost:11434"
    device: str = "cpu"
    compute_type: str = "int8"
    capture_samplerate: int = 48000
    samplerate: int = 16000
    block_ms: int = 100
    chunk_sec: float = 6.0
    min_rms: float = 0.003
    input_device: int | str | None = None
    output_device: str | None = None
    duration_sec: float | None = None
    source: str = "both"
    vad_filter: bool = True
    no_speech_threshold: float = 0.35
    gain: float = 1.0
    ai_correct: bool = False
    ai_provider: str = "ollama"
    ai_mode: str = "explain"
    ai_model: str = "deepseek-v3.1:671b-cloud"
    ai_base_url: str = ""
    ai_api_key: str = ""
    history_turns: int = 6
    ai_timeout: float = 60.0
    debug_stt: bool = False
    auto_recover: bool = True
    ai_auto_tune: bool = True


def now_text() -> str:
    return datetime.now().strftime("%H:%M:%S")


def parse_device(value: str | None) -> int | str | None:
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def is_meaningful_text(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return False
    korean_count = sum(1 for char in compact if "\uac00" <= char <= "\ud7a3")
    japanese_count = sum(1 for char in compact if "\u3040" <= char <= "\u30ff")
    chinese_count = sum(1 for char in compact if "\u4e00" <= char <= "\u9fff")
    alpha_count = len(re.findall(r"[A-Za-z0-9]", compact))
    signal_count = korean_count + japanese_count + chinese_count + alpha_count
    if signal_count < 2:
        return False
    return signal_count / max(len(compact), 1) >= 0.25


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.size / float(src_rate)
    dst_size = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_size, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def print_devices() -> None:
    print(sd.query_devices())
    try:
        import soundcard as sc

        print()
        print("System output devices for loopback capture:")
        for index, speaker in enumerate(sc.all_speakers()):
            print(f"[system {index}] {speaker.name}")
    except Exception as exc:
        print()
        print(f"System output device list unavailable: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime Korean microphone/system speech-to-text")
    parser.add_argument("--list-devices", action="store_true", help="show audio devices and exit")
    parser.add_argument("--self-test", action="store_true", help="check imports and audio devices, then exit")
    parser.add_argument("--model-test", action="store_true", help="load the Whisper model, then exit")
    parser.add_argument("--input-device", help="input device index or name substring")
    parser.add_argument("--output-device", help="system speaker/output device name substring for loopback capture")
    parser.add_argument("--source", choices=("mic", "system", "both"), default="both")
    parser.add_argument("--model", default="small", help="Whisper model size: tiny, base, small, medium, large-v3")
    parser.add_argument("--stt-engine", choices=("whisper", "ollama"), default="whisper")
    parser.add_argument("--ollama-stt-model", default=os.getenv("OLLAMA_STT_MODEL", "karanchopda333/whisper"))
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--capture-samplerate", type=int, default=48000, help="audio capture sample rate")
    parser.add_argument("--samplerate", type=int, default=16000, help="Whisper transcription sample rate")
    parser.add_argument("--chunk-sec", type=float, default=6.0)
    parser.add_argument("--min-rms", type=float, default=0.003)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--vad-filter", action=argparse.BooleanOptionalAction, default=True, help="enable faster-whisper VAD filter")
    parser.add_argument("--no-speech-threshold", type=float, default=0.35)
    parser.add_argument("--duration-sec", type=float, help="run live mode for N seconds, then exit")
    parser.add_argument("--ai-correct", action="store_true")
    parser.add_argument("--ai-provider", choices=("ollama", "openai"), default=os.getenv("AI_PROVIDER", "ollama"))
    parser.add_argument("--ai-mode", choices=("correct", "complete", "explain", "chat", "both"), default="complete")
    parser.add_argument("--ai-model", default=os.getenv("AI_MODEL", "deepseek-v3.1:671b-cloud"))
    parser.add_argument(
        "--ai-base-url",
        default=os.getenv("AI_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")),
    )
    parser.add_argument("--ai-api-key", default=os.getenv("AI_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")))
    parser.add_argument("--history-turns", type=int, default=6, help="recent transcript chunks used by AI completion")
    parser.add_argument("--ai-timeout", type=float, default=60.0, help="AI request timeout in seconds")
    parser.add_argument("--debug-stt", action="store_true", help="print discarded Whisper text candidates")
    parser.add_argument(
        "--auto-recover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="automatically retry STT with safer settings when speech is missed",
    )
    parser.add_argument(
        "--ai-auto-tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="let the configured AI model choose STT recovery settings from telemetry",
    )
    return parser


def load_model(cfg: SttConfig) -> WhisperModel:
    print(f"[{now_text()}] Loading model: {cfg.model_size} / {cfg.device} / {cfg.compute_type}")
    print(f"[{now_text()}] First run can download the model. Please wait.")
    model = WhisperModel(cfg.model_size, device=cfg.device, compute_type=cfg.compute_type)
    print(f"[{now_text()}] Model ready.")
    return model


def wav_base64(audio: np.ndarray, samplerate: int) -> str:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(samplerate)
        wav.writeframes(pcm16.tobytes())
    return base64.b64encode(bio.getvalue()).decode("ascii")


def transcribe_with_ollama(audio: np.ndarray, cfg: SttConfig) -> str | None:
    payload = {
        "model": cfg.ollama_stt_model,
        "messages": [
        {
            "role": "user",
            "content": (
                "Transcribe this audio. Accept Korean, English, Japanese, and Chinese. "
                "Return only the spoken text in the original language."
            ),
            "audio": [wav_base64(audio, cfg.samplerate)],
        }
        ],
        "stream": False,
    }
    url = cfg.ollama_base_url.rstrip("/") + "/api/chat"
    try:
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        result = json.loads(body)
        message = result.get("message", {})
        content = str(message.get("content", "")).strip()
        return content if is_meaningful_text(content) else None
    except HTTPError as exc:
        print(f"[{now_text()}] ollama-stt> request failed: HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        return None
    except (URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        print(f"[{now_text()}] ollama-stt> request failed: {exc}", file=sys.stderr)
        return None


def collect_meaningful_segments(segments, cfg: SttConfig, label: str) -> list[str]:
    texts: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if is_meaningful_text(text):
            texts.append(text)
        elif cfg.debug_stt and text:
            print(f"[{now_text()}] whisper debug discarded {label}> {text}")
    return texts


def run_whisper_pass(
    model: WhisperModel,
    audio: np.ndarray,
    cfg: SttConfig,
    *,
    language: str | None,
    vad_filter: bool,
    no_speech_threshold: float,
    condition_on_previous_text: bool,
    label: str,
) -> list[str]:
    segments, _info = model.transcribe(
        audio,
        language=language,
        task="transcribe",
        beam_size=8,
        best_of=5,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
        no_speech_threshold=no_speech_threshold,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-0.75,
    )
    return collect_meaningful_segments(segments, cfg, label)


def transcribe_with_whisper(model: WhisperModel, audio: np.ndarray, cfg: SttConfig) -> list[str]:
    texts = run_whisper_pass(
        model,
        audio,
        cfg,
        language=None,
        vad_filter=cfg.vad_filter,
        no_speech_threshold=cfg.no_speech_threshold,
        condition_on_previous_text=True,
        label="normal",
    )
    if texts or not cfg.auto_recover:
        return texts

    recovered = run_whisper_pass(
        model,
        audio,
        cfg,
        language=None,
        vad_filter=False,
        no_speech_threshold=max(cfg.no_speech_threshold, 0.75),
        condition_on_previous_text=False,
        label="auto-ko",
    )
    if recovered:
        print(f"[{now_text()}] auto recover> STT recovered with relaxed multilingual settings")
        return recovered

    recovered = run_whisper_pass(
        model,
        audio,
        cfg,
        language=None,
        vad_filter=False,
        no_speech_threshold=max(cfg.no_speech_threshold, 0.80),
        condition_on_previous_text=False,
        label="auto-any",
    )
    if recovered:
        print(f"[{now_text()}] auto recover> STT recovered with automatic language detection")
    return recovered


def load_story_requirements() -> str:
    path = os.getenv("LYRIC_STORY_BOARD_REQUIREMENTS_PATH", "").strip()
    if not path:
        return ""

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""

    return str(data.get("story_requirements", "")).strip()


def format_runtime_context(history: list[str], cfg: SttConfig) -> str:
    recent = "\n".join(history[-cfg.history_turns :])
    story_requirements = load_story_requirements()
    story_rule = (
        "\nLYRIC STORY BOARD USER REQUIREMENTS\n"
        f"{story_requirements}\n"
        "- When filling story_1, story_2, story_3, lyric_theme, scene, conflict, emotional_arc, and related lyric fields, stay inside these requirements.\n"
        "- If the user says they do not know the detailed backstory, explain or infer only as a marked suggestion, and keep it consistent with these requirements.\n"
        if story_requirements
        else ""
    )
    return (
        "SYSTEM CONTEXT\n"
        "- Topic: infer the current topic from recent USER_MIC lines; keep it stable until the user changes topic.\n"
        "- Content: use the latest recognized speech and the recent context.\n"
        "- Place: Voice Context Console, a live desktop microphone and video-sound session.\n"
        "- Location: local PC audio environment; infer a scene/location from speech only when it is clearly mentioned.\n"
        "- Time: current real-time session.\n"
        "- Doing: listen to USER_MIC and VIDEO_SOUND separately; help the user continuously write song lyrics.\n"
        "- Source rule: USER_MIC is the user's own microphone. VIDEO_SOUND is PC/browser/game/video playback audio.\n"
        "- Language rule: recognized speech may be Korean, English, Japanese, or Chinese; AI replies and all structured fields must be Korean.\n"
        "- Creative mode: the user is making song lyrics. Preserve lyric ideas, story threads, mood, imagery, hook ideas, and next-line candidates.\n"
        f"RECENT CONTEXT\n{recent or '(none)'}"
        f"{story_rule}"
    )


def build_ai_messages(
    text: str,
    cfg: SttConfig,
    source_label: str,
    history: list[str],
) -> list[dict[str, str]]:
    context = format_runtime_context(history, cfg)
    source_desc = (
        "USER_MIC: the user's own live microphone speech. Reply directly to the user."
        if source_label == "mic"
        else "VIDEO_SOUND: audio from a video, game, browser, or PC playback. Do not treat it as the user's own words."
    )
    if cfg.ai_mode == "correct":
        system_prompt = (
            "You correct multilingual speech-to-text output. Accept Korean, English, Japanese, and Chinese. "
            "Return only one natural Korean sentence that preserves the meaning. Do not explain."
        )
        user_prompt = f"{context}\n\nSource meaning: {source_desc}\nCorrect and translate this speech recognition result into natural Korean:\n{text}"
    elif cfg.ai_mode == "complete":
        system_prompt = (
            "You are a Korean live sentence completion assistant. "
            "The input is a short, possibly broken speech-to-text fragment. "
            "The fragment may be Korean, English, Japanese, or Chinese. "
            "Always use the SYSTEM CONTEXT when it helps. "
            "Return one natural completed Korean sentence. "
            "Do not add facts that were not implied. Do not explain."
        )
        user_prompt = (
            f"{context}\n\n"
            f"Source meaning: {source_desc}\n"
            f"Current fragment:\n{text}\n\n"
            "Complete it as one natural Korean sentence."
        )
    elif cfg.ai_mode == "chat":
        system_prompt = (
            "You are a real-time Korean lyric-writing assistant inside a Voice Context Console. "
            "Always reference the SYSTEM CONTEXT and organize the current situation. "
            "Keep USER_MIC and VIDEO_SOUND separate. "
            "Recognized speech may be Korean, English, Japanese, or Chinese, but every JSON value must be Korean. Translate meaning into Korean when needed. "
            "The user is continuously creating song lyrics, so maintain story threads, lyric themes, emotional arcs, imagery, hooks, and next line candidates. "
            "If the source is USER_MIC, reply naturally and help continue the lyric work. "
            "If the source is VIDEO_SOUND, treat it as reference material or background audio; summarize it for lyric inspiration and do not answer as if the user said it. "
            "Never invent people, places, reasons, relationships, or events that are not present in the recognized text or recent context. "
            "For creative lyric fields, you may propose ideas, but mark them as suggestions and do not present them as factual context. "
            "If the speech is short, cut off, unclear, or ambiguous, mark speech_state as incomplete or unclear and use '-' for factual unknown fields. "
            "Evidence must quote or closely paraphrase only the recognized text. "
            "Return only one strict JSON object. Do not wrap it in markdown. "
            "All values must be Korean strings. "
            "Use '-' when unknown. "
            "Schema: {"
            "\"reply\":\"short chatbot reply\","
            "\"topic\":\"current topic\","
            "\"content\":\"current content summary\","
            "\"place\":\"current place or scene\","
            "\"location\":\"more specific location/source\","
            "\"time\":\"when this is happening\","
            "\"doing\":\"what is happening now\","
            "\"who\":\"who is involved\","
            "\"what\":\"what happened or what is being discussed\","
            "\"where\":\"where it happens\","
            "\"when\":\"when it happens\","
            "\"why\":\"why it happens or likely reason\","
            "\"how\":\"how it happens or tone/method\","
            "\"intent\":\"speaker intent\","
            "\"target\":\"target object/person/topic\","
            "\"scene\":\"scene or situation\","
            "\"emotion\":\"emotion or tone\","
            "\"speech_state\":\"complete, incomplete, unclear, noise, or mixed\","
            "\"confidence\":\"high, medium, or low\","
            "\"evidence\":\"short evidence from recognized text\","
            "\"assumption\":\"what was inferred; '-' if none\","
            "\"next_action\":\"what the chatbot should do next\","
            "\"story_1\":\"surface story or immediate lyric situation\","
            "\"story_2\":\"hidden emotional story\","
            "\"story_3\":\"world, scene, or background story\","
            "\"lyric_theme\":\"main lyric theme\","
            "\"lyric_mood\":\"mood palette\","
            "\"genre_hint\":\"genre or style hint\","
            "\"narrator\":\"speaker or narrator of the song\","
            "\"listener\":\"who the song is addressed to\","
            "\"conflict\":\"inner or outer conflict\","
            "\"emotional_arc\":\"emotional movement\","
            "\"key_images\":\"visual images separated by comma\","
            "\"symbol_words\":\"symbolic words separated by comma\","
            "\"hook_idea\":\"chorus or hook idea\","
            "\"verse_idea\":\"verse idea\","
            "\"pre_chorus_idea\":\"pre-chorus idea\","
            "\"bridge_idea\":\"bridge idea\","
            "\"next_line_1\":\"next lyric line candidate\","
            "\"next_line_2\":\"next lyric line candidate\","
            "\"next_line_3\":\"next lyric line candidate\","
            "\"rhyme_words\":\"rhyme or sound-alike words separated by comma\","
            "\"keywords\":\"important keywords separated by comma\","
            "\"questions\":\"useful questions for the songwriter\","
            "\"source_summary\":\"USER_MIC or VIDEO_SOUND summary\""
            "}."
        )
        user_prompt = (
            f"{context}\n\n"
            f"Source meaning: {source_desc}\n"
            f"Recognized text:\n{text}\n\n"
            "Analyze and respond using the JSON schema."
        )
    elif cfg.ai_mode == "both":
        system_prompt = (
            "You are a Korean live transcript assistant. Accept Korean, English, Japanese, and Chinese input. Return two short lines in Korean: "
            "first 'Completed:' with a natural completed sentence, then 'Explanation:' with a concise explanation."
        )
        user_prompt = f"{context}\n\nSource meaning: {source_desc}\nKorean speech recognition result:\n{text}"
    else:
        system_prompt = (
            "You are a Korean live commentary assistant. Explain or summarize the given multilingual live transcript "
            "in Korean. Be very concise. Return one short Korean sentence only."
        )
        user_prompt = f"{context}\n\nSource meaning: {source_desc}\nKorean live transcript:\n{text}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def run_ai_ollama(messages: list[dict[str, str]], cfg: SttConfig) -> str | None:
    payload = {
        "model": cfg.ai_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1100},
    }
    url = cfg.ollama_base_url.rstrip("/") + "/api/chat"
    try:
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=cfg.ai_timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        result = json.loads(body)
        return str(result.get("message", {}).get("content", "")).strip() or None
    except HTTPError as exc:
        print(f"[{now_text()}] ollama-ai> request failed: HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        return None
    except (URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        print(f"[{now_text()}] ollama-ai> request failed: {exc}", file=sys.stderr)
        return None


def run_ai_openai_compatible(messages: list[dict[str, str]], cfg: SttConfig) -> str | None:
    if not cfg.ai_base_url:
        print(f"[{now_text()}] ai> skipped: AI_BASE_URL is empty", file=sys.stderr)
        return None
    headers = {"Content-Type": "application/json"}
    if cfg.ai_api_key:
        headers["Authorization"] = f"Bearer {cfg.ai_api_key}"
    elif "localhost" not in cfg.ai_base_url and "127.0.0.1" not in cfg.ai_base_url:
        print(f"[{now_text()}] ai> skipped: set DEEPSEEK_API_KEY or AI_API_KEY", file=sys.stderr)
        return None
    payload = {
        "model": cfg.ai_model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1400,
        "stream": False,
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req = Request(cfg.ai_base_url, data=data, headers=headers, method="POST")
        with urlopen(req, timeout=cfg.ai_timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        result = json.loads(body)
        return result["choices"][0]["message"]["content"].strip()
    except HTTPError as exc:
        if exc.code == 401:
            cfg.ai_correct = False
            print(
                f"[{now_text()}] ai> unauthorized: disabling AI completion for this run. "
                "Check API key/base URL.",
                file=sys.stderr,
            )
        else:
            print(f"[{now_text()}] ai> request failed: HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        return None
    except (URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
        print(f"[{now_text()}] ai> request failed: {exc}", file=sys.stderr)
        return None


def run_ai(
    text: str,
    cfg: SttConfig,
    source_label: str,
    history: list[str],
) -> str | None:
    if not cfg.ai_correct or not text.strip():
        return None
    messages = build_ai_messages(text, cfg, source_label, history)
    if cfg.ai_provider == "ollama":
        return run_ai_ollama(messages, cfg)
    return run_ai_openai_compatible(messages, cfg)


def parse_ai_context(text: str) -> dict[str, str] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    wanted = (
        "reply",
        "topic",
        "content",
        "place",
        "location",
        "time",
        "doing",
        "who",
        "what",
        "where",
        "when",
        "why",
        "how",
        "intent",
        "target",
        "scene",
        "emotion",
        "speech_state",
        "confidence",
        "evidence",
        "assumption",
        "next_action",
        "story_1",
        "story_2",
        "story_3",
        "lyric_theme",
        "lyric_mood",
        "genre_hint",
        "narrator",
        "listener",
        "conflict",
        "emotional_arc",
        "key_images",
        "symbol_words",
        "hook_idea",
        "verse_idea",
        "pre_chorus_idea",
        "bridge_idea",
        "next_line_1",
        "next_line_2",
        "next_line_3",
        "rhyme_words",
        "keywords",
        "questions",
        "source_summary",
    )
    return {key: str(data.get(key, "-")).strip() or "-" for key in wanted}


AI_STT_PROFILES: dict[str, dict[str, float | bool]] = {
    "balanced": {"vad_filter": False, "no_speech_threshold": 0.70, "gain": 1.0},
    "sensitive": {"vad_filter": False, "no_speech_threshold": 0.85, "gain": 1.35},
    "noisy": {"vad_filter": True, "no_speech_threshold": 0.50, "gain": 0.9},
    "loud_system": {"vad_filter": False, "no_speech_threshold": 0.65, "gain": 0.75},
}


def parse_json_object(text: str) -> dict[str, object] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def ask_ai_for_stt_profile(
    cfg: SttConfig,
    source_label: str,
    level: float,
    miss_streak: int,
) -> tuple[str, str] | None:
    messages = [
        {
            "role": "system",
            "content": (
                "You tune a live Korean STT pipeline. Choose exactly one profile "
                "from: balanced, sensitive, noisy, loud_system. Return only JSON "
                "with keys profile and reason. Use sensitive when speech is quiet "
                "or repeatedly missed, noisy when low-quality noise causes false "
                "attempts, loud_system when PC/system audio is loud, balanced for "
                "normal recovery."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "source": source_label,
                    "rms": round(level, 5),
                    "miss_streak": miss_streak,
                    "current": {
                        "vad_filter": cfg.vad_filter,
                        "no_speech_threshold": cfg.no_speech_threshold,
                        "gain": cfg.gain,
                        "min_rms": cfg.min_rms,
                        "chunk_sec": cfg.chunk_sec,
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]
    ai_text = run_ai_ollama(messages, cfg) if cfg.ai_provider == "ollama" else run_ai_openai_compatible(messages, cfg)
    if not ai_text:
        return None
    data = parse_json_object(ai_text)
    if not data:
        return None
    profile = str(data.get("profile", "")).strip().lower()
    reason = str(data.get("reason", "")).strip()
    if profile not in AI_STT_PROFILES:
        return None
    return profile, reason or "AI selected this recovery profile."


def fallback_stt_profile(source_label: str, level: float, miss_streak: int) -> tuple[str, str]:
    if source_label == "system" and level >= 0.08:
        return "loud_system", "fallback: loud system audio was repeatedly missed"
    if miss_streak >= 6 or level < 0.01:
        return "sensitive", "fallback: repeated misses or quiet speech"
    return "balanced", "fallback: normal recovery"


def apply_stt_profile(cfg: SttConfig, profile: str, reason: str, *, by_ai: bool) -> None:
    values = AI_STT_PROFILES[profile]
    cfg.vad_filter = bool(values["vad_filter"])
    cfg.no_speech_threshold = float(values["no_speech_threshold"])
    cfg.gain = float(values["gain"])
    actor = "AI auto tune" if by_ai else "auto tune"
    print(
        f"[{now_text()}] {actor}> profile={profile} "
        f"vad_filter={cfg.vad_filter} no_speech_threshold={cfg.no_speech_threshold:.2f} "
        f"gain={cfg.gain:.2f} reason={reason}"
    )


def start_ai_worker(
    cfg: SttConfig,
    ai_queue: queue.Queue[tuple[str, str, list[str]]],
    stop_event: threading.Event,
) -> threading.Thread:
    def worker() -> None:
        while not stop_event.is_set():
            try:
                source_label, raw_text, history_snapshot = ai_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            ai_text = run_ai(raw_text, cfg, source_label, history_snapshot)
            if ai_text:
                if cfg.ai_mode == "chat":
                    context = parse_ai_context(ai_text)
                    if context:
                        context["source"] = source_label
                        print(f"[{now_text()}] context ai> {json.dumps(context, ensure_ascii=False)}")
                        print(f"[{now_text()}] {source_label} deepseek> {context.get('reply', '-')}")
                    else:
                        print(f"[{now_text()}] {source_label} deepseek> {ai_text}")
                else:
                    label = "complete" if cfg.ai_mode == "complete" else "deepseek"
                    print(f"[{now_text()}] {source_label} {label}> {ai_text}")
            ai_queue.task_done()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def pick_speaker(name_part: str | None):
    import soundcard as sc

    speakers = sc.all_speakers()
    if not speakers:
        raise RuntimeError("No system output devices found")
    if name_part:
        lowered = name_part.lower()
        for speaker in speakers:
            if lowered in speaker.name.lower():
                return speaker
        raise RuntimeError(f"System output device not found: {name_part}")
    return sc.default_speaker()


def start_system_loopback(
    cfg: SttConfig,
    audio_queue: queue.Queue[tuple[str, np.ndarray]],
    capture_blocksize: int,
    stop_event: threading.Event,
) -> threading.Thread:
    def worker() -> None:
        try:
            import soundcard as sc
            from soundcard.mediafoundation import SoundcardRuntimeWarning

            warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)
            speaker = pick_speaker(cfg.output_device)
            loopback = sc.get_microphone(speaker.name, include_loopback=True)
            print(f"[{now_text()}] system source> {speaker.name}")
            with loopback.recorder(samplerate=cfg.capture_samplerate, channels=1) as recorder:
                while not stop_event.is_set():
                    data = recorder.record(numframes=capture_blocksize)
                    mono = np.mean(data, axis=1).astype(np.float32, copy=True)
                    try:
                        audio_queue.put_nowait(("system", mono))
                    except queue.Full:
                        pass
        except Exception as exc:
            print(f"[{now_text()}] system source failed: {exc}", file=sys.stderr)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def transcribe_loop(cfg: SttConfig) -> int:
    audio_queue: queue.Queue[tuple[str, np.ndarray]] = queue.Queue(maxsize=20)
    capture_blocksize = int(cfg.capture_samplerate * cfg.block_ms / 1000)
    chunk_samples = int(cfg.capture_samplerate * cfg.chunk_sec)
    model = None if cfg.stt_engine == "ollama" else load_model(cfg)

    def callback(indata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        mono = np.mean(indata, axis=1).astype(np.float32, copy=True)
        try:
            audio_queue.put_nowait(("mic", mono))
        except queue.Full:
            pass

    print()
    print("========================================")
    print(" Korean Mic STT - live mode")
    print("========================================")
    print(f"Audio source: {cfg.source}")
    print(f"STT engine: {cfg.stt_engine}")
    if cfg.stt_engine == "ollama":
        print(f"Ollama STT model: {cfg.ollama_stt_model}")
        print(f"Ollama URL: {cfg.ollama_base_url}")
    print(f"Capture rate: {cfg.capture_samplerate} Hz -> Whisper rate: {cfg.samplerate} Hz")
    if cfg.source in ("mic", "both"):
        print("Mic input: ON")
    if cfg.source in ("system", "both"):
        print("System sound input: ON")
    print("Audio is processed in RAM chunks. No recording file is created.")
    if cfg.ai_correct:
        print(f"AI: ON / provider={cfg.ai_provider} / mode={cfg.ai_mode} / model={cfg.ai_model}")
    print("Press Ctrl+C to stop.")
    print()

    buffers: dict[str, np.ndarray] = {}
    chunk_indexes: dict[str, int] = {}
    miss_streaks: dict[str, int] = {}
    global_history: list[str] = []
    stop_at = time.monotonic() + cfg.duration_sec if cfg.duration_sec else None
    stop_event = threading.Event()
    ai_queue: queue.Queue[tuple[str, str, list[str]]] = queue.Queue(maxsize=4)
    if cfg.ai_correct:
        start_ai_worker(cfg, ai_queue, stop_event)

    try:
        with ExitStack() as stack:
            if cfg.source in ("mic", "both"):
                stack.enter_context(
                    sd.InputStream(
                        samplerate=cfg.capture_samplerate,
                        blocksize=capture_blocksize,
                        dtype="float32",
                        channels=1,
                        device=cfg.input_device,
                        callback=callback,
                    )
                )
            if cfg.source in ("system", "both"):
                start_system_loopback(cfg, audio_queue, capture_blocksize, stop_event)

            while True:
                if stop_at is not None and time.monotonic() >= stop_at:
                    stop_event.set()
                    print(f"[{now_text()}] Duration test complete.")
                    return 0

                source_label, block = audio_queue.get()
                buffer = np.concatenate((buffers.get(source_label, np.empty(0, dtype=np.float32)), block))
                if buffer.size > chunk_samples * 3:
                    buffer = buffer[-chunk_samples * 2 :]
                if buffer.size < chunk_samples:
                    buffers[source_label] = buffer
                    continue

                capture_audio = buffer[:chunk_samples]
                buffers[source_label] = buffer[chunk_samples:]
                if cfg.gain != 1.0:
                    capture_audio = np.clip(capture_audio * cfg.gain, -1.0, 1.0)
                level = rms(capture_audio)
                chunk_index = chunk_indexes.get(source_label, 0) + 1
                chunk_indexes[source_label] = chunk_index

                if level < cfg.min_rms:
                    print(f"[{now_text()}] {source_label}> #{chunk_index:04d} silence rms={level:.4f}")
                    continue

                audio = resample_linear(capture_audio, cfg.capture_samplerate, cfg.samplerate)
                print(f"[{now_text()}] {source_label}> #{chunk_index:04d} transcribing rms={level:.4f} ...")
                if cfg.stt_engine == "ollama":
                    ollama_text = transcribe_with_ollama(audio, cfg)
                    texts = [ollama_text] if ollama_text else []
                else:
                    if model is None:
                        raise RuntimeError("Whisper model was not loaded")
                    texts = transcribe_with_whisper(model, audio, cfg)
                if texts:
                    raw_text = " ".join(texts)
                    miss_streaks[source_label] = 0
                    print(f"[{now_text()}] {source_label} {cfg.stt_engine}> {raw_text}")
                    history_label = "USER_MIC" if source_label == "mic" else "VIDEO_SOUND"
                    if cfg.ai_correct:
                        try:
                            ai_queue.put_nowait((source_label, raw_text, list(global_history)))
                        except queue.Full:
                            print(f"[{now_text()}] ai> busy: skipped commentary for latest chunk")
                    global_history.append(f"{history_label}: {raw_text}")
                    del global_history[:-cfg.history_turns]
                else:
                    miss_streak = miss_streaks.get(source_label, 0) + 1
                    miss_streaks[source_label] = miss_streak
                    print(f"[{now_text()}] {source_label}> #{chunk_index:04d} no supported speech detected")
                    if cfg.auto_recover and cfg.ai_auto_tune and miss_streak >= 3 and miss_streak % 3 == 0:
                        advice = ask_ai_for_stt_profile(cfg, source_label, level, miss_streak)
                        if advice:
                            profile, reason = advice
                            apply_stt_profile(cfg, profile, reason, by_ai=True)
                        else:
                            profile, reason = fallback_stt_profile(source_label, level, miss_streak)
                            print(f"[{now_text()}] AI auto tune> AI unavailable or invalid response; using fallback")
                            apply_stt_profile(cfg, profile, reason, by_ai=False)
    except KeyboardInterrupt:
        stop_event.set()
        print("\nStopped.")
        return 0
    except Exception as exc:
        stop_event.set()
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Run with --list-devices to check microphone/system output devices.", file=sys.stderr)
        return 1


def self_test() -> int:
    print(f"[{now_text()}] Self-test started.")
    devices = sd.query_devices()
    input_count = sum(1 for device in devices if int(device.get("max_input_channels", 0)) > 0)
    try:
        import soundcard as sc

        speaker_count = len(sc.all_speakers())
        print(f"[{now_text()}] soundcard OK.")
        print(f"[{now_text()}] System output devices found: {speaker_count}")
    except Exception as exc:
        speaker_count = 0
        print(f"[{now_text()}] soundcard/system loopback warning: {exc}")
    print(f"[{now_text()}] sounddevice OK.")
    print(f"[{now_text()}] faster-whisper OK.")
    print(f"[{now_text()}] Input devices found: {input_count}")
    if input_count <= 0:
        print(f"[{now_text()}] ERROR: No microphone/input device found.", file=sys.stderr)
        return 1
    if speaker_count <= 0:
        print(f"[{now_text()}] WARNING: No system output device found for loopback capture.")
    print(f"[{now_text()}] Self-test passed.")
    return 0


def config_from_args(args: argparse.Namespace) -> SttConfig:
    return SttConfig(
        model_size=args.model,
        stt_engine=args.stt_engine,
        ollama_stt_model=args.ollama_stt_model,
        ollama_base_url=args.ollama_base_url,
        capture_samplerate=args.capture_samplerate,
        samplerate=args.samplerate,
        chunk_sec=args.chunk_sec,
        min_rms=args.min_rms,
        input_device=parse_device(args.input_device),
        output_device=args.output_device,
        duration_sec=args.duration_sec,
        source=args.source,
        vad_filter=args.vad_filter,
        no_speech_threshold=args.no_speech_threshold,
        gain=args.gain,
        ai_correct=args.ai_correct,
        ai_provider=args.ai_provider,
        ai_mode=args.ai_mode,
        ai_model=args.ai_model,
        ai_base_url=args.ai_base_url,
        ai_api_key=args.ai_api_key,
        history_turns=args.history_turns,
        ai_timeout=args.ai_timeout,
        debug_stt=args.debug_stt,
        auto_recover=args.auto_recover,
        ai_auto_tune=args.ai_auto_tune,
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.list_devices:
        print_devices()
        return 0
    if args.self_test:
        return self_test()
    cfg = config_from_args(args)
    if args.model_test:
        load_model(cfg)
        print(f"[{now_text()}] Model-test passed.")
        return 0
    return transcribe_loop(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
