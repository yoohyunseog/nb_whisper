from __future__ import annotations

import argparse
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "voice_game.html"
SETTINGS_DIR = ROOT / "settings"
STORY_REQUIREMENTS_PATH = SETTINGS_DIR / "lyric_story_board_requirements.json"
EVENT_RE = re.compile(r"^\[(?P<time>[^\]]+)\]\s+(?P<source>mic|system)\s+(?P<kind>whisper|deepseek|complete|whisper|ollama)>\s+(?P<text>.+)$")
STATE_RE = re.compile(r"^\[(?P<time>[^\]]+)\]\s+(?P<source>mic|system)>\s+#(?P<num>\d+)\s+(?P<state>transcribing|silence|no Korean speech detected|no supported speech detected)(?:\s+rms=(?P<rms>[0-9.]+))?")
CONTEXT_RE = re.compile(r"^\[(?P<time>[^\]]+)\]\s+context ai>\s+(?P<data>\{.+\})$")


class EventHub:
    def __init__(self) -> None:
        self.clients: list[queue.Queue[dict[str, str]]] = []
        self.lock = threading.Lock()
        self.recent: list[dict[str, str]] = []

    def subscribe(self) -> queue.Queue[dict[str, str]]:
        client: queue.Queue[dict[str, str]] = queue.Queue(maxsize=100)
        with self.lock:
            self.clients.append(client)
            for event in self.recent[-20:]:
                self._put(client, event)
        return client

    def unsubscribe(self, client: queue.Queue[dict[str, str]]) -> None:
        with self.lock:
            if client in self.clients:
                self.clients.remove(client)

    def publish(self, event: dict[str, str]) -> None:
        with self.lock:
            self.recent.append(event)
            del self.recent[:-50]
            for client in list(self.clients):
                self._put(client, event)

    @staticmethod
    def _put(client: queue.Queue[dict[str, str]], event: dict[str, str]) -> None:
        try:
            client.put_nowait(event)
        except queue.Full:
            try:
                client.get_nowait()
                client.put_nowait(event)
            except queue.Empty:
                pass


def parse_line(line: str) -> dict[str, str] | None:
    line = line.strip()
    if not line:
        return None

    match = CONTEXT_RE.match(line)
    if match:
        data = match.groupdict()
        try:
            context = json.loads(data["data"])
        except json.JSONDecodeError:
            context = {"content": data["data"]}
        if not isinstance(context, dict):
            context = {"content": str(context)}
        source = str(context.get("source", "system"))
        return {
            "type": "context",
            "source": source,
            "sourceLabel": "User Mic" if source == "mic" else "Video Sound",
            "time": data["time"],
            **context,
        }

    match = EVENT_RE.match(line)
    if match:
        data = match.groupdict()
        kind = data["kind"]
        return {
            "type": "speech" if kind in {"whisper", "ollama"} else "ai",
            "source": data["source"],
            "sourceLabel": "User Mic" if data["source"] == "mic" else "Video Sound",
            "kind": kind,
            "text": data["text"],
            "time": data["time"],
        }

    match = STATE_RE.match(line)
    if match:
        data = match.groupdict()
        return {
            "type": "state",
            "source": data["source"],
            "sourceLabel": "User Mic" if data["source"] == "mic" else "Video Sound",
            "state": data["state"],
            "rms": data.get("rms") or "",
            "time": data["time"],
        }

    if "ollama-ai> request failed" in line or "ai>" in line:
        return {"type": "system", "source": "system", "text": line, "time": time.strftime("%H:%M:%S")}

    return None


def run_stt(hub: EventHub, args: argparse.Namespace) -> None:
    pyexe = ROOT / ".venv_stt" / "Scripts" / "python.exe"
    if not pyexe.exists():
        pyexe = Path(sys.executable)

    cmd = [
        str(pyexe),
        "-u",
        str(ROOT / "korean_mic_stt.py"),
        "--model",
        args.model,
        "--source",
        args.source,
        "--chunk-sec",
        str(args.chunk_sec),
        "--min-rms",
        str(args.min_rms),
        "--no-speech-threshold",
        str(args.no_speech_threshold),
        "--ai-correct",
        "--ai-provider",
        "ollama",
        "--ai-mode",
        args.ai_mode,
        "--ai-model",
        args.ai_model,
        "--ai-timeout",
        str(args.ai_timeout),
    ]
    if args.gain is not None:
        cmd.extend(["--gain", str(args.gain)])
    if args.duration_sec is not None:
        cmd.extend(["--duration-sec", str(args.duration_sec)])
    if args.vad_filter:
        cmd.append("--vad-filter")
    else:
        cmd.append("--no-vad-filter")
    if args.auto_recover:
        cmd.append("--auto-recover")
    else:
        cmd.append("--no-auto-recover")
    if args.ai_auto_tune:
        cmd.append("--ai-auto-tune")
    else:
        cmd.append("--no-ai-auto-tune")
    if args.debug_stt:
        cmd.append("--debug-stt")
    if args.extra:
        cmd.extend(args.extra)

    hub.publish({"type": "system", "source": "system", "text": "voice engine starting", "time": time.strftime("%H:%M:%S")})
    env = os.environ.copy()
    env["LYRIC_STORY_BOARD_REQUIREMENTS_PATH"] = str(STORY_REQUIREMENTS_PATH)

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        event = parse_line(line)
        if event:
            hub.publish(event)
    hub.publish({"type": "system", "source": "system", "text": f"voice engine stopped: {proc.poll()}", "time": time.strftime("%H:%M:%S")})


def make_handler(hub: EventHub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
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

            if parsed.path == "/story-requirements":
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                value = ""
                if STORY_REQUIREMENTS_PATH.exists():
                    try:
                        data = json.loads(STORY_REQUIREMENTS_PATH.read_text(encoding="utf-8"))
                        value = str(data.get("story_requirements", ""))
                    except (OSError, json.JSONDecodeError):
                        value = ""
                body = json.dumps({"story_requirements": value}, ensure_ascii=False)
                self.wfile.write(body.encode("utf-8"))
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
                            data = json.dumps(event, ensure_ascii=False)
                            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                        except queue.Empty:
                            self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, socket.timeout):
                    hub.unsubscribe(client)
                return

            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/story-requirements":
                self.send_response(404)
                self.end_headers()
                return

            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                data = {}

            SETTINGS_DIR.mkdir(exist_ok=True)
            STORY_REQUIREMENTS_PATH.write_text(
                json.dumps(
                    {"story_requirements": str(data.get("story_requirements", ""))},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="small")
    parser.add_argument("--source", choices=("mic", "system", "both"), default="both")
    parser.add_argument("--chunk-sec", type=float, default=6.0)
    parser.add_argument("--min-rms", type=float, default=0.003)
    parser.add_argument("--vad-filter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-speech-threshold", type=float, default=0.35)
    parser.add_argument("--auto-recover", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ai-auto-tune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-stt", action="store_true")
    parser.add_argument("--ai-mode", choices=("correct", "complete", "explain", "chat", "both"), default="chat")
    parser.add_argument("--ai-model", default="deepseek-v3.1:671b-cloud")
    parser.add_argument("--ai-timeout", type=float, default=60.0)
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    hub = EventHub()
    worker = threading.Thread(target=run_stt, args=(hub, args), daemon=True)
    worker.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(hub))
    print(f"Voice game server: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
