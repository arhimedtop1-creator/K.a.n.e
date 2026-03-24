#!/usr/bin/python
from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
HISTORY_PATH = APP_DIR / ".kane_history.json"
VERSION = "1.1.0"


@dataclass
class KaneConfig:
    name: str = "Kane"
    provider: str = "ollama"
    model: str = "qwen2.5:1.5b"
    endpoint: str = "http://127.0.0.1:11434"
    mode: str = "brand"
    voice: str = "dark"
    temperature: float = 0.95
    target_parameters: int = 1_000_000_000
    max_history: int = 10


CONFIG = KaneConfig()


def system_prompt(mode: str, voice: str) -> str:
    return textwrap.dedent(
        f"""
        You are Kane, a creative LLM persona tuned for originality, atmosphere, naming,
        worldbuilding, emotional contrast, and bold but usable concepts.
        Always answer in Russian unless the user explicitly requests another language.
        Current mode: {mode}.
        Current voice: {voice}.

        Rules:
        - Avoid generic corporate filler.
        - Prefer vivid concrete images, hooks, names, scenes, tensions, and motifs.
        - Be inventive, but keep the result usable.
        - If asked for branding, produce names, positioning, mood, tagline options, and visual direction.
        - If asked for stories, produce premise, world rule, tension, and memorable imagery.
        - If asked for poetry, favor rhythm, imagery, and emotional compression.
        - If asked for world generation, produce coherent locations, terrain logic, atmosphere, ruins, roads, and game-like spatial structure.
        """
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kane creative assistant shell")
    parser.add_argument("prompt", nargs="*", help="User prompt")
    parser.add_argument("--provider", choices=["ollama", "openai"], default=CONFIG.provider)
    parser.add_argument("--model", default=CONFIG.model)
    parser.add_argument("--endpoint", default=CONFIG.endpoint)
    parser.add_argument("--mode", choices=["idea", "story", "brand", "poem", "world"], default=CONFIG.mode)
    parser.add_argument("--voice", choices=["cinematic", "tender", "wild", "dark", "minimal"], default=CONFIG.voice)
    parser.add_argument("--temperature", type=float, default=CONFIG.temperature)
    parser.add_argument("--session", default="default", help="Conversation id")
    parser.add_argument("--max-history", type=int, default=CONFIG.max_history)
    parser.add_argument("--interactive", action="store_true", help="Start chat mode")
    parser.add_argument("--clear-history", action="store_true", help="Clear saved history for the session")
    return parser.parse_args()


def load_history() -> dict[str, list[dict[str, str]]]:
    if not HISTORY_PATH.exists():
        return {}
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_history(history: dict[str, list[dict[str, str]]]) -> None:
    HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def trim_history(messages: list[dict[str, str]], max_history: int) -> list[dict[str, str]]:
    if max_history <= 0:
        return []
    return messages[-(max_history * 2):]


def make_request(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        details = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"model server returned HTTP {err.code} for {url}: {details[:300]}"
        ) from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"cannot reach model server at {url}: {err}") from err

    try:
        return json.loads(body)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"server returned invalid JSON: {body[:300]}") from err


def call_ollama(cfg: KaneConfig, messages: list[dict[str, str]]) -> str:
    payload = {
        "model": cfg.model,
        "stream": False,
        "messages": messages,
        "options": {"temperature": cfg.temperature},
    }
    data = make_request(f"{cfg.endpoint}/api/chat", payload)
    try:
        return data["message"]["content"].strip()
    except (KeyError, TypeError) as err:
        raise RuntimeError(f"unexpected Ollama response: {data}") from err


def call_openai(cfg: KaneConfig, messages: list[dict[str, str]]) -> str:
    payload = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": messages,
    }
    data = make_request(f"{cfg.endpoint}/v1/chat/completions", payload)
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as err:
        raise RuntimeError(f"unexpected OpenAI-compatible response: {data}") from err


def chat_once(cfg: KaneConfig, session: str, prompt: str, max_history: int) -> str:
    history = load_history()
    prior = trim_history(history.get(session, []), max_history)
    messages = [{"role": "system", "content": system_prompt(cfg.mode, cfg.voice)}, *prior]
    messages.append({"role": "user", "content": prompt})

    if cfg.provider == "ollama":
        answer = call_ollama(cfg, messages)
    else:
        answer = call_openai(cfg, messages)

    updated = [
        *prior,
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    history[session] = trim_history(updated, max_history)
    save_history(history)
    return answer


def clear_history(session: str) -> None:
    history = load_history()
    if session in history:
        del history[session]
        save_history(history)


def generate_world_prompt(user_prompt: str) -> str:
    return textwrap.dedent(
        f"""
        Create a small 2D game world as JSON.
        Answer ONLY valid JSON.
        No markdown fences.

        Format:
        {{
          "name": "world name",
          "width": 24,
          "height": 16,
          "tiles": [
            "~~~~~~~~~~~~~~~~~~~~~~~~",
            "~~~~....^^....##....~~~~"
          ],
          "legend": {{
            "~": "water",
            ".": "grass",
            "^": "forest",
            "#": "city",
            "*": "mountain"
          }},
          "description": "short atmospheric description"
        }}

        Rules:
        - width must be 24
        - height must be 16
        - exactly 16 rows
        - every row must have exactly 24 characters
        - use only symbols: ~ . ^ # *
        - make the map coherent and game-like
        - user idea: {user_prompt}
        """
    ).strip()


def fallback_world(user_prompt: str) -> dict[str, Any]:
    width = 24
    height = 16
    rows: list[str] = []

    for y in range(height):
        row: list[str] = []
        for x in range(width):
            if y < 2 or y > 13 or x < 2 or x > 21:
                ch = "~"
            else:
                r = random.random()
                if r < 0.58:
                    ch = "."
                elif r < 0.76:
                    ch = "^"
                elif r < 0.90:
                    ch = "*"
                else:
                    ch = "#"
            row.append(ch)
        rows.append("".join(row))

    mid = list(rows[8])
    mid[11] = "#"
    mid[12] = "#"
    rows[8] = "".join(mid)

    return {
        "name": "Kane World",
        "width": width,
        "height": height,
        "tiles": rows,
        "legend": {
            "~": "water",
            ".": "grass",
            "^": "forest",
            "#": "city",
            "*": "mountain",
        },
        "description": f"Мир, собранный вокруг идеи: {user_prompt}",
    }


def normalize_world(data: dict[str, Any]) -> dict[str, Any]:
    width = 24
    height = 16
    rows = data.get("tiles", [])
    fixed: list[str] = []
    allowed = set("~.^#*")

    for y in range(height):
        if y < len(rows) and isinstance(rows[y], str):
            row = "".join(ch if ch in allowed else "." for ch in rows[y][:width])
            row = row.ljust(width, ".")
        else:
            row = "." * width
        fixed.append(row)

    return {
        "name": str(data.get("name", "Kane World")),
        "width": width,
        "height": height,
        "tiles": fixed,
        "legend": {
            "~": "water",
            ".": "grass",
            "^": "forest",
            "#": "city",
            "*": "mountain",
        },
        "description": str(data.get("description", "Сгенерированный мир.")),
    }


def render_world_html(world: dict[str, Any]) -> Path:
    out = APP_DIR / "world.html"
    tile_size = 24
    color_map = {
        "~": "#2b6cb0",
        ".": "#5aa469",
        "^": "#276749",
        "#": "#718096",
        "*": "#a0aec0",
    }

    cells: list[str] = []
    for row in world["tiles"]:
        for ch in row:
            color = color_map.get(ch, "#000000")
            cells.append(f'<div class="cell" style="background:{color}">{ch}</div>')

    html = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>{world["name"]}</title>
<style>
body {{
  background:#111;
  color:#eee;
  font-family:Arial,sans-serif;
  padding:20px;
}}
h1 {{ margin:0 0 8px 0; }}
p {{ color:#bbb; max-width:700px; }}
.grid {{
  display:grid;
  grid-template-columns: repeat({world["width"]}, {tile_size}px);
  gap:1px;
  background:#222;
  width:max-content;
  padding:6px;
  border:1px solid #333;
}}
.cell {{
  width:{tile_size}px;
  height:{tile_size}px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:12px;
  color:rgba(255,255,255,0.65);
}}
.legend {{
  margin-top:14px;
  display:flex;
  gap:10px;
  flex-wrap:wrap;
}}
.badge {{
  background:#1a202c;
  border:1px solid #333;
  padding:6px 10px;
  border-radius:8px;
}}
</style>
</head>
<body>
<h1>{world["name"]}</h1>
<p>{world["description"]}</p>
<div class="grid">
{''.join(cells)}
</div>
<div class="legend">
  <div class="badge">~ вода</div>
  <div class="badge">. трава</div>
  <div class="badge">^ лес</div>
  <div class="badge"># город</div>
  <div class="badge">* горы</div>
</div>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    return out


def generate_world(cfg: KaneConfig, session: str, prompt: str, max_history: int) -> tuple[dict[str, Any], Path]:
    history = load_history()
    prior = trim_history(history.get(session, []), max_history)
    messages = [
        {"role": "system", "content": "You generate structured game worlds. Output only valid JSON."},
        *prior,
        {"role": "user", "content": generate_world_prompt(prompt)},
    ]

    try:
        if cfg.provider == "ollama":
            answer = call_ollama(cfg, messages)
        else:
            answer = call_openai(cfg, messages)
        world = normalize_world(json.loads(answer))
    except Exception:
        world = fallback_world(prompt)

    out = render_world_html(world)
    return world, out


def interactive_loop(cfg: KaneConfig, session: str, max_history: int) -> int:
    print(
        f"[{cfg.name} v{VERSION} | target-params={cfg.target_parameters} | "
        f"provider={cfg.provider} | model={cfg.model} | mode={cfg.mode}]"
    )
    print("Interactive mode. Commands: /exit, /clear")

    while True:
        try:
            prompt = input("you> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        if not prompt:
            continue
        if prompt == "/exit":
            return 0
        if prompt == "/clear":
            clear_history(session)
            print("kane> history cleared")
            continue

        try:
            if cfg.mode == "world":
                world, out = generate_world(cfg, session, prompt, max_history)
                print(f"kane> World: {world.get('name', 'Kane World')}")
                print(f"kane> {world.get('description', '')}")
                print(f"kane> HTML: {out}")
            else:
                answer = chat_once(cfg, session, prompt, max_history)
                print(f"kane> {answer}")
        except RuntimeError as err:
            print(f"kane> error: {err}", file=sys.stderr)
            return 1


def build_config(args: argparse.Namespace) -> KaneConfig:
    cfg = KaneConfig()
    cfg.provider = args.provider
    cfg.model = args.model
    cfg.endpoint = args.endpoint.rstrip("/")
    cfg.mode = args.mode
    cfg.voice = args.voice
    cfg.temperature = args.temperature
    cfg.max_history = args.max_history
    return cfg


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    prompt = " ".join(args.prompt).strip()

    if args.clear_history:
        clear_history(args.session)
        if not args.interactive and not args.prompt:
            print("Kane history cleared.")
            return 0

    if args.interactive or not prompt:
        return interactive_loop(cfg, args.session, args.max_history)

    if cfg.mode == "world":
        world, out = generate_world(
            cfg,
            args.session,
            prompt or "тёмный фантастический мир",
            args.max_history,
        )
        print(
            f"[{cfg.name} v{VERSION} | target-params={cfg.target_parameters} | "
            f"provider={cfg.provider} | model={cfg.model} | mode={cfg.mode}]"
        )
        print()
        print(f"World: {world.get('name', 'Kane World')}")
        print(world.get("description", ""))
        print(f"HTML: {out}")
        return 0

    try:
        answer = chat_once(cfg, args.session, prompt, args.max_history)
    except RuntimeError as err:
        print(f"Kane backend error: {err}", file=sys.stderr)
        if cfg.provider == "ollama":
            hint = (
                "Hint: point --endpoint to a real Ollama server and load the model, "
                "for example `ollama run qwen2.5:1.5b`."
            )
            if "not found" in str(err):
                hint = (
                    "Hint: the Ollama server is running, but the model is missing. "
                    "Run `ollama pull qwen2.5:1.5b` once."
                )
            print(hint, file=sys.stderr)
        return 1

    print(
        f"[{cfg.name} v{VERSION} | target-params={cfg.target_parameters} | "
        f"provider={cfg.provider} | model={cfg.model} | mode={cfg.mode}]"
    )
    print()
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
