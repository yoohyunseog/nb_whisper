from __future__ import annotations

import argparse
import json
import queue
import re
import socket
import sys
import threading
import time
import wave
import warnings
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "voice_game.html"
SETTINGS_DIR = ROOT / "settings"
STORY_REQUIREMENTS_PATH = SETTINGS_DIR / "lyric_story_board_requirements.json"
TRANSCRIPT_PREFIX = "Lyric Story Board user requirements"


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8765
    model: str = "medium"
    source: str = "both"
    input_device: str | int | None = None
    output_device: str | None = None
    device: str = "cuda"
    compute_type: str = "float16"
    capture_samplerate: int = 48000
    whisper_samplerate: int = 16000
    block_ms: int = 100
    chunk_sec: float = 15.0
    min_rms: float = 0.003
    vad_filter: bool = True
    no_speech_threshold: float = 0.45
    ai_provider: str = "ollama"
    ai_model: str = "deepseek-v3.1:671b-cloud"
    ollama_base_url: str = "http://localhost:11434"
    ai_base_url: str = ""
    ai_api_key: str = ""
    ai_timeout: float = 45.0
    no_ai: bool = False


class EventHub:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.clients: list[queue.Queue[dict[str, str]]] = []
        self.recent: list[dict[str, str]] = []

    def publish(self, event: dict[str, str]) -> None:
        with self.lock:
            self.recent.append(event)
            del self.recent[:-100]
            for client in list(self.clients):
                try:
                    client.put_nowait(event)
                except queue.Full:
                    try:
                        client.get_nowait()
                        client.put_nowait(event)
                    except queue.Empty:
                        pass

    def subscribe(self) -> queue.Queue[dict[str, str]]:
        client: queue.Queue[dict[str, str]] = queue.Queue(maxsize=100)
        with self.lock:
            self.clients.append(client)
            for event in self.recent[-30:]:
                client.put_nowait(event)
        return client

    def unsubscribe(self, client: queue.Queue[dict[str, str]]) -> None:
        with self.lock:
            if client in self.clients:
                self.clients.remove(client)


def now_text() -> str:
    return datetime.now().strftime("%H:%M:%S")


def parse_device(value: str | None) -> int | str | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.size / float(src_rate)
    dst_size = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_size, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def is_supported_text(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return False
    korean_count = sum(1 for char in compact if "\uac00" <= char <= "\ud7a3")
    alpha_count = len(re.findall(r"[A-Za-z0-9]", compact))
    return korean_count >= 2 or alpha_count >= 2


def list_devices() -> None:
    print(sd.query_devices())
    try:
        import soundcard as sc

        print()
        print("System output devices for Windows loopback:")
        for index, speaker in enumerate(sc.all_speakers()):
            print(f"[system {index}] {speaker.name}")
    except Exception as exc:
        print(f"System output device list unavailable: {exc}")


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


def capture_system(cfg: Config, audio_queue: queue.Queue[tuple[str, np.ndarray]], stop_event: threading.Event, hub: EventHub) -> None:
    try:
        import soundcard as sc
        from soundcard.mediafoundation import SoundcardRuntimeWarning

        warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)
        speaker = pick_speaker(cfg.output_device)
        loopback = sc.get_microphone(speaker.name, include_loopback=True)
        hub.publish({"type": "system", "source": "system", "text": f"system source: {speaker.name}", "time": now_text()})
        print(f"[{now_text()}] system source> {speaker.name}")
        blocksize = int(cfg.capture_samplerate * cfg.block_ms / 1000)
        with loopback.recorder(samplerate=cfg.capture_samplerate, channels=1) as recorder:
            while not stop_event.is_set():
                data = recorder.record(numframes=blocksize)
                mono = np.mean(data, axis=1).astype(np.float32, copy=True)
                try:
                    audio_queue.put_nowait(("system", mono))
                except queue.Full:
                    try:
                        audio_queue.get_nowait()
                        audio_queue.put_nowait(("system", mono))
                    except queue.Empty:
                        pass
    except Exception as exc:
        text = f"system source failed: {exc}"
        hub.publish({"type": "system", "source": "system", "text": text, "time": now_text()})
        print(f"[{now_text()}] {text}", file=sys.stderr)


def transcribe(model: WhisperModel, audio: np.ndarray, cfg: Config) -> str:
    def collect_text(vad_filter: bool, no_speech_threshold: float, beam_size: int, best_of: int) -> str:
        segments, _info = model.transcribe(
            audio,
            task="transcribe",
            beam_size=beam_size,
            best_of=best_of,
            vad_filter=vad_filter,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=False,
        )
        texts = [segment.text.strip() for segment in segments if is_supported_text(segment.text)]
        return " ".join(texts).strip()

    text = collect_text(
        vad_filter=cfg.vad_filter,
        no_speech_threshold=cfg.no_speech_threshold,
        beam_size=5,
        best_of=5,
    )
    if text:
        return text

    return collect_text(
        vad_filter=False,
        no_speech_threshold=0.9,
        beam_size=1,
        best_of=1,
    )


def is_cuda_runtime_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "cublas" in text or "cudnn" in text or "cuda" in text


def run_ollama(messages: list[dict[str, str]], cfg: Config) -> str | None:
    payload = {"model": cfg.ai_model, "messages": messages, "stream": False, "options": {"temperature": 0.2}}
    url = cfg.ollama_base_url.rstrip("/") + "/api/chat"
    try:
        req = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=cfg.ai_timeout) as resp:
            result = json.loads(resp.read().decode("utf-8", errors="replace"))
        return str(result.get("message", {}).get("content", "")).strip() or None
    except (HTTPError, URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        print(f"[{now_text()}] ai> ollama failed: {exc}", file=sys.stderr)
        return None


def run_openai_compatible(messages: list[dict[str, str]], cfg: Config) -> str | None:
    if not cfg.ai_base_url:
        return None
    headers = {"Content-Type": "application/json"}
    if cfg.ai_api_key:
        headers["Authorization"] = f"Bearer {cfg.ai_api_key}"
    payload = {"model": cfg.ai_model, "messages": messages, "temperature": 0.2, "stream": False}
    try:
        req = Request(cfg.ai_base_url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urlopen(req, timeout=cfg.ai_timeout) as resp:
            result = json.loads(resp.read().decode("utf-8", errors="replace"))
        return result["choices"][0]["message"]["content"].strip()
    except (HTTPError, URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
        print(f"[{now_text()}] ai> openai-compatible failed: {exc}", file=sys.stderr)
        return None


CONTEXT_KEYS = (
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


def parse_context_json(text: str) -> dict[str, str] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()
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
    return {key: str(data.get(key, "-")).strip() or "-" for key in CONTEXT_KEYS}


def load_story_requirements() -> str:
    if not STORY_REQUIREMENTS_PATH.exists():
        return ""
    try:
        data = json.loads(STORY_REQUIREMENTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("story_requirements", "")).strip()


def format_transcript_with_requirements(text: str) -> str:
    story_requirements = load_story_requirements()
    if not story_requirements:
        return text
    return f"{TRANSCRIPT_PREFIX}: {story_requirements}\n{text}"


def build_context_with_ai(source: str, text: str, history: list[str], cfg: Config) -> dict[str, str] | None:
    if cfg.no_ai:
        return None
    source_desc = "USER_MIC is the user's own microphone." if source == "mic" else "VIDEO_SOUND is PC/system playback audio."
    messages = [
        {
            "role": "system",
            "content": (
                "You are a real-time Korean lyric-writing assistant inside a Voice Context Console. "
                "Always answer in Korean. Keep USER_MIC and VIDEO_SOUND separate. "
                "The user is continuously creating song lyrics, so maintain story threads, lyric themes, emotional arcs, imagery, hooks, and next line candidates. "
                "Never invent factual people, places, reasons, relationships, or events that are not present in the transcript or recent context. "
                "For creative lyric fields, you may propose ideas, but keep them consistent with the transcript and mark unknown factual fields with '-'. "
                "Return only one strict JSON object. Do not wrap it in markdown. "
                "All values must be Korean strings. Use '-' when unknown. "
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
            ),
        },
        {
            "role": "user",
            "content": (
                f"{source_desc}\n\nRecent context:\n" + "\n".join(history[-8:]) +
                f"\n\nRecognized Korean transcript:\n{text}\n\nAnalyze and respond using the JSON schema."
            ),
        },
    ]
    ai_text = run_openai_compatible(messages, cfg) if cfg.ai_provider == "openai" else run_ollama(messages, cfg)
    if not ai_text:
        return None
    return parse_context_json(ai_text)


def build_lyrics_with_ai(story_board: dict[str, str], prompt: str, cfg: Config, story_requirements: str | None = None) -> str | None:
    if cfg.no_ai:
        return None
    story_requirements = story_requirements if story_requirements is not None else load_story_requirements()
    storyboard_lines = [
        f"{key}: {value}"
        for key, value in story_board.items()
        if str(value or "").strip() and str(value or "").strip() != "-"
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Korean lyric-writing assistant. "
                "Write original Korean song lyrics from the provided Lyric Story Board. "
                "Use the Story Board as the source of truth for story, narrator, listener, conflict, images, symbols, hook, and emotional arc. "
                "Follow the user's extra prompt when it does not contradict the Story Board. "
                "Do not copy existing songs or quote copyrighted lyrics. "
                "Return only the finished lyrics with clear section labels such as [Verse 1], [Pre-Chorus], [Chorus], [Bridge], [Outro]."
            ),
        },
        {
            "role": "user",
            "content": (
                "Lyric Story Board user requirements:\n"
                f"{story_requirements or '-'}\n\n"
                "Current Lyric Story Board:\n"
                f"{chr(10).join(storyboard_lines) or '-'}\n\n"
                "Additional lyric prompt:\n"
                f"{prompt.strip() or '-'}\n\n"
                "이 정보를 보고 바로 부를 수 있는 한국어 가사를 만들어줘."
            ),
        },
    ]
    return run_openai_compatible(messages, cfg) if cfg.ai_provider == "openai" else run_ollama(messages, cfg)


def ai_worker(cfg: Config, hub: EventHub, ai_queue: queue.Queue[tuple[str, str, list[str]]], stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            source, text, history = ai_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        context = build_context_with_ai(source, text, history, cfg)
        if context:
            context["type"] = "context"
            context["source"] = source
            context["time"] = now_text()
            hub.publish(context)
            reply = context.get("reply", "-")
            hub.publish({"type": "ai", "source": source, "text": reply, "time": now_text()})
            print(f"[{now_text()}] context ai> {json.dumps(context, ensure_ascii=False)}")
            print(f"[{now_text()}] {source} ai> {reply}")
        ai_queue.task_done()


def stt_worker(cfg: Config, hub: EventHub, stop_event: threading.Event) -> None:
    audio_queue: queue.Queue[tuple[str, np.ndarray]] = queue.Queue(maxsize=240)
    ai_queue: queue.Queue[tuple[str, str, list[str]]] = queue.Queue(maxsize=8)
    buffers: dict[str, np.ndarray] = {}
    indexes: dict[str, int] = {}
    history: list[str] = []

    print(f"[{now_text()}] loading faster-whisper model: {cfg.model} / {cfg.device} / {cfg.compute_type}")
    hub.publish(
        {
            "type": "system",
            "source": "system",
            "text": f"loading model: {cfg.model} / {cfg.device} / {cfg.compute_type}",
            "time": now_text(),
        }
    )
    try:
        model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
    except RuntimeError as exc:
        if cfg.device == "cuda":
            text = f"GPU load failed, falling back to CPU int8: {exc}"
            print(f"[{now_text()}] {text}", file=sys.stderr)
            hub.publish({"type": "system", "source": "system", "text": text, "time": now_text()})
            cfg.device = "cpu"
            cfg.compute_type = "int8"
            if cfg.model not in {"tiny", "base", "small"}:
                cfg.model = "small"
                hub.publish({"type": "system", "source": "system", "text": "CPU fallback changed model to small for realtime capture", "time": now_text()})
            model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
        else:
            raise
    hub.publish({"type": "system", "source": "system", "text": "model ready", "time": now_text()})
    print(f"[{now_text()}] model ready")

    threading.Thread(target=ai_worker, args=(cfg, hub, ai_queue, stop_event), daemon=True).start()

    def mic_callback(indata: np.ndarray, _frames: int, _time, status) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        mono = np.mean(indata, axis=1).astype(np.float32, copy=True)
        try:
            audio_queue.put_nowait(("mic", mono))
        except queue.Full:
            try:
                audio_queue.get_nowait()
                audio_queue.put_nowait(("mic", mono))
            except queue.Empty:
                pass

    capture_blocksize = int(cfg.capture_samplerate * cfg.block_ms / 1000)
    chunk_samples = int(cfg.capture_samplerate * cfg.chunk_sec)

    with ExitStack() as stack:
        if cfg.source in ("mic", "both"):
            stack.enter_context(
                sd.InputStream(
                    samplerate=cfg.capture_samplerate,
                    blocksize=capture_blocksize,
                    dtype="float32",
                    channels=1,
                    device=cfg.input_device,
                    latency="high",
                    callback=mic_callback,
                )
            )
            hub.publish({"type": "system", "source": "mic", "text": "mic source started", "time": now_text()})
            print(f"[{now_text()}] mic source> started")

        if cfg.source in ("system", "both"):
            threading.Thread(target=capture_system, args=(cfg, audio_queue, stop_event, hub), daemon=True).start()

        while not stop_event.is_set():
            source, block = audio_queue.get()
            buffer = np.concatenate((buffers.get(source, np.empty(0, dtype=np.float32)), block))
            if buffer.size > chunk_samples * 3:
                buffer = buffer[-chunk_samples * 2 :]
            if buffer.size < chunk_samples:
                buffers[source] = buffer
                continue

            capture = buffer[:chunk_samples]
            buffers[source] = buffer[chunk_samples:]
            level = rms(capture)
            indexes[source] = indexes.get(source, 0) + 1
            index = indexes[source]

            if level < cfg.min_rms:
                hub.publish({"type": "state", "source": source, "state": "silence", "rms": f"{level:.4f}", "text": f"silence rms={level:.4f}", "time": now_text()})
                print(f"[{now_text()}] {source}> #{index:04d} silence rms={level:.4f}")
                continue

            hub.publish({"type": "state", "source": source, "state": "transcribing", "rms": f"{level:.4f}", "text": f"transcribing rms={level:.4f}", "time": now_text()})
            print(f"[{now_text()}] {source}> #{index:04d} transcribing rms={level:.4f} ...")
            audio = resample_linear(capture, cfg.capture_samplerate, cfg.whisper_samplerate)
            try:
                text = transcribe(model, audio, cfg)
            except RuntimeError as exc:
                if cfg.device == "cuda" and is_cuda_runtime_error(exc):
                    message = f"GPU transcribe failed, switching to CPU int8: {exc}"
                    print(f"[{now_text()}] {message}", file=sys.stderr)
                    hub.publish({"type": "system", "source": "system", "text": message, "time": now_text()})
                    cfg.device = "cpu"
                    cfg.compute_type = "int8"
                    if cfg.model not in {"tiny", "base", "small"}:
                        cfg.model = "small"
                        hub.publish({"type": "system", "source": "system", "text": "CPU fallback changed model to small for realtime capture", "time": now_text()})
                    model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
                    text = transcribe(model, audio, cfg)
                else:
                    raise
            if not text:
                hub.publish({"type": "state", "source": source, "state": "no supported speech detected", "rms": f"{level:.4f}", "text": "no supported speech detected", "time": now_text()})
                print(f"[{now_text()}] {source}> #{index:04d} no supported speech detected")
                continue

            text = format_transcript_with_requirements(text)
            label = "USER_MIC" if source == "mic" else "VIDEO_SOUND"
            history.append(f"{label}: {text}")
            del history[:-12]
            event = {"type": "transcript", "source": source, "text": text, "time": now_text()}
            hub.publish(event)
            print(f"[{now_text()}] {source} whisper> {text}")
            try:
                ai_queue.put_nowait((source, text, list(history)))
            except queue.Full:
                print(f"[{now_text()}] ai> busy, skipped summary")


def make_handler(hub: EventHub, cfg: Config):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _fmt: str, *_args) -> None:
            return

        def handle(self) -> None:
            try:
                super().handle()
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, socket.timeout):
                return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(HTML_PATH.read_bytes())
                return
            if parsed.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                client = hub.subscribe()
                try:
                    while True:
                        try:
                            event = client.get(timeout=5)
                            self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
                        except queue.Empty:
                            self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, socket.timeout):
                    hub.unsubscribe(client)
                return
            if parsed.path == "/story-requirements":
                value = load_story_requirements()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(json.dumps({"story_requirements": value}, ensure_ascii=False).encode("utf-8"))
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path not in {"/story-requirements", "/generate-lyrics"}:
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                data = {}
            if parsed.path == "/generate-lyrics":
                if cfg.no_ai:
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(json.dumps({"ok": False, "error": "AI is disabled with --no-ai."}).encode("utf-8"))
                    return
                story_board = data.get("story_board", {})
                if not isinstance(story_board, dict):
                    story_board = {}
                prompt = str(data.get("prompt", ""))
                story_requirements = str(data.get("story_requirements", ""))
                lyrics = build_lyrics_with_ai({str(k): str(v) for k, v in story_board.items()}, prompt, cfg, story_requirements)
                if not lyrics:
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(json.dumps({"ok": False, "error": "AI lyric generation failed."}).encode("utf-8"))
                    return
                hub.publish({"type": "ai", "source": "bot", "text": "가사를 생성했습니다.", "time": now_text()})
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "lyrics": lyrics}, ensure_ascii=False).encode("utf-8"))
                return
            SETTINGS_DIR.mkdir(exist_ok=True)
            STORY_REQUIREMENTS_PATH.write_text(
                json.dumps({"story_requirements": str(data.get("story_requirements", ""))}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual source faster-whisper STT server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--source", choices=("mic", "system", "both"), default="both")
    parser.add_argument("--input-device")
    parser.add_argument("--output-device")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--chunk-sec", type=float, default=15.0)
    parser.add_argument("--min-rms", type=float, default=0.003)
    parser.add_argument("--vad-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-speech-threshold", type=float, default=0.45)
    parser.add_argument("--ai-provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--ai-model", default="deepseek-v3.1:671b-cloud")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--ai-base-url", default="")
    parser.add_argument("--ai-api-key", default="")
    parser.add_argument("--ai-timeout", type=float, default=45.0)
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        host=args.host,
        port=args.port,
        model=args.model,
        source=args.source,
        input_device=parse_device(args.input_device),
        output_device=args.output_device,
        device=args.device,
        compute_type=args.compute_type,
        chunk_sec=args.chunk_sec,
        min_rms=args.min_rms,
        vad_filter=args.vad_filter,
        no_speech_threshold=args.no_speech_threshold,
        ai_provider=args.ai_provider,
        ai_model=args.ai_model,
        ollama_base_url=args.ollama_base_url,
        ai_base_url=args.ai_base_url,
        ai_api_key=args.ai_api_key,
        ai_timeout=args.ai_timeout,
        no_ai=args.no_ai,
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.list_devices:
        list_devices()
        return 0

    cfg = config_from_args(args)
    hub = EventHub()
    stop_event = threading.Event()
    threading.Thread(target=stt_worker, args=(cfg, hub, stop_event), daemon=True).start()
    server = ThreadingHTTPServer((cfg.host, cfg.port), make_handler(hub, cfg))
    print(f"Voice dual STT server: http://{cfg.host}:{cfg.port}")
    print(f"Source: {cfg.source} / chunk-sec={cfg.chunk_sec} / model={cfg.model}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop_event.set()
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
