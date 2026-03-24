#!/usr/bin/python
from __future__ import annotations

import argparse
import json
import random
import textwrap
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
STATE_PATH = APP_DIR / "kane_state.json"


@dataclass
class KaneConfig:
    model: str = "qwen2.5:1.5b"
    endpoint: str = "http://127.0.0.1:11434"
    temperature: float = 0.8
    port: int = 8000
    tick_seconds: float = 2.0
    prompt: str = "пустой мрачный мир, который Kane должен сам развить"


COLOR_MAP = {
    "~": "#2b6cb0",
    ".": "#5aa469",
    "^": "#276749",
    "*": "#9aa0a6",
}

BUILDING_TYPES = {
    "hut": {"symbol": "H", "color": "#8b5e3c", "label": "хижина"},
    "tower": {"symbol": "T", "color": "#c0c0c0", "label": "башня"},
    "fire": {"symbol": "F", "color": "#ff6b35", "label": "костёр"},
    "ruin": {"symbol": "R", "color": "#6b7280", "label": "руины"},
    "garden": {"symbol": "G", "color": "#2ecc71", "label": "сад"},
}


def make_request(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        details = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {err.code}: {details[:300]}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"cannot reach model server: {err}") from err

    return json.loads(body)


def call_ollama(cfg: KaneConfig, messages: list[dict[str, str]]) -> str:
    payload = {
        "model": cfg.model,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": cfg.temperature,
        },
    }
    data = make_request(f"{cfg.endpoint.rstrip('/')}/api/chat", payload)
    return data["message"]["content"].strip()


def create_empty_world() -> dict[str, Any]:
    width = 32
    height = 20
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                row.append("~")
            else:
                row.append(".")
        rows.append("".join(row))

    return {
        "name": "Kane Colony",
        "description": "Пустой мир, который Kane развивает сам.",
        "width": width,
        "height": height,
        "tiles": rows,
    }


def initial_state(cfg: KaneConfig) -> dict[str, Any]:
    if STATE_PATH.exists():
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if "world" in data and "kane" in data:
                return data
        except Exception:
            pass

    world = create_empty_world()
    return {
        "world": world,
        "tick": 0,
        "time_of_day": "night",
        "kane": {
            "x": world["width"] // 2,
            "y": world["height"] // 2,
            "mood": "curious",
            "energy": 100,
            "last_action": "SPAWN",
            "last_words": "Я начну с пустоты.",
        },
        "buildings": [],
        "log": [
            "Kane появился в пустом мире.",
            f"Замысел: {cfg.prompt}",
        ],
    }


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def tile_at(state: dict[str, Any], x: int, y: int) -> str:
    world = state["world"]
    if x < 0 or y < 0 or x >= world["width"] or y >= world["height"]:
        return "~"
    return world["tiles"][y][x]


def building_at(state: dict[str, Any], x: int, y: int) -> dict[str, Any] | None:
    for b in state["buildings"]:
        if b["x"] == x and b["y"] == y:
            return b
    return None


def can_walk(state: dict[str, Any], x: int, y: int) -> bool:
    if tile_at(state, x, y) == "~":
        return False
    return True


def can_build(state: dict[str, Any], x: int, y: int) -> bool:
    if tile_at(state, x, y) == "~":
        return False
    if building_at(state, x, y) is not None:
        return False
    if state["kane"]["x"] == x and state["kane"]["y"] == y:
        return False
    return True


def nearby_summary(state: dict[str, Any]) -> dict[str, str]:
    x = state["kane"]["x"]
    y = state["kane"]["y"]

    def fmt(xx: int, yy: int) -> str:
        b = building_at(state, xx, yy)
        if b:
            return BUILDING_TYPES[b["type"]]["label"]
        t = tile_at(state, xx, yy)
        return {
            "~": "вода",
            ".": "трава",
            "^": "лес",
            "*": "камень",
        }.get(t, "неизвестно")

    return {
        "north": fmt(x, y - 1),
        "south": fmt(x, y + 1),
        "west": fmt(x - 1, y),
        "east": fmt(x + 1, y),
        "here": fmt(x, y),
    }


def llm_decision_prompt(state: dict[str, Any]) -> str:
    kane = state["kane"]
    near = nearby_summary(state)
    recent_log = state["log"][-8:]

    return textwrap.dedent(
        f"""
        Ты Kane. Ты живёшь в мире и развиваешь его.
        Отвечай ТОЛЬКО JSON, без markdown.

        Состояние:
        - tick: {state["tick"]}
        - время: {state["time_of_day"]}
        - позиция: ({kane["x"]}, {kane["y"]})
        - настроение: {kane["mood"]}
        - энергия: {kane["energy"]}
        - построек уже: {len(state["buildings"])}

        Рядом:
        - север: {near["north"]}
        - юг: {near["south"]}
        - запад: {near["west"]}
        - восток: {near["east"]}
        - здесь: {near["here"]}

        Последние события:
        {json.dumps(recent_log, ensure_ascii=False)}

        Выбери одно действие:
        - MOVE_NORTH
        - MOVE_SOUTH
        - MOVE_WEST
        - MOVE_EAST
        - WAIT
        - BUILD_HUT
        - BUILD_TOWER
        - BUILD_FIRE
        - BUILD_RUIN
        - BUILD_GARDEN

        Формат:
        {{
          "action": "BUILD_HUT",
          "say": "короткая фраза на русском",
          "mood": "одно слово"
        }}

        Правила:
        - не иди в воду
        - строй мир постепенно
        - иногда двигайся, иногда строй
        - отвечай только JSON
        """
    ).strip()


def fallback_decision(state: dict[str, Any]) -> dict[str, str]:
    kane = state["kane"]
    x = kane["x"]
    y = kane["y"]

    walk_options = []
    for action, nx, ny in [
        ("MOVE_NORTH", x, y - 1),
        ("MOVE_SOUTH", x, y + 1),
        ("MOVE_WEST", x - 1, y),
        ("MOVE_EAST", x + 1, y),
    ]:
        if can_walk(state, nx, ny):
            walk_options.append(action)

    build_options = []
    if state["tick"] % 3 == 0:
        for building_action in ["BUILD_HUT", "BUILD_TOWER", "BUILD_FIRE", "BUILD_RUIN", "BUILD_GARDEN"]:
            if has_free_neighbor(state):
                build_options.append(building_action)

    pool = walk_options + build_options + ["WAIT"]
    if not pool:
        action = "WAIT"
    else:
        action = random.choice(pool)

    phrases = {
        "MOVE_NORTH": "Север всегда обещает больше, чем должен.",
        "MOVE_SOUTH": "С юга приходит тёплый пепел.",
        "MOVE_WEST": "Запад хранит недостроенные формы.",
        "MOVE_EAST": "На востоке мир звучит точнее.",
        "WAIT": "Иногда миру нужно дать загустеть.",
        "BUILD_HUT": "Пусть здесь появится первое укрытие.",
        "BUILD_TOWER": "Миру нужна вертикаль.",
        "BUILD_FIRE": "Начнём с огня.",
        "BUILD_RUIN": "Даже у нового мира должна быть древность.",
        "BUILD_GARDEN": "Жизнь должна укорениться.",
    }

    return {
        "action": action,
        "say": phrases.get(action, "Я продолжаю."),
        "mood": random.choice(["curious", "focused", "dark", "restless"]),
    }


def ask_kane(cfg: KaneConfig, state: dict[str, Any]) -> dict[str, str]:
    messages = [
        {
            "role": "system",
            "content": "Ты Kane, агент-строитель мира. Отвечай только JSON.",
        },
        {
            "role": "user",
            "content": llm_decision_prompt(state),
        },
    ]
    try:
        answer = call_ollama(cfg, messages)
        data = json.loads(answer)
        action = str(data.get("action", "WAIT"))
        say = str(data.get("say", ""))
        mood = str(data.get("mood", "curious"))
        allowed = {
            "MOVE_NORTH", "MOVE_SOUTH", "MOVE_WEST", "MOVE_EAST",
            "WAIT",
            "BUILD_HUT", "BUILD_TOWER", "BUILD_FIRE", "BUILD_RUIN", "BUILD_GARDEN",
        }
        if action not in allowed:
            return fallback_decision(state)
        return {
            "action": action,
            "say": say[:100] or "Я продолжаю.",
            "mood": mood[:24] or "curious",
        }
    except Exception:
        return fallback_decision(state)


def has_free_neighbor(state: dict[str, Any]) -> bool:
    x = state["kane"]["x"]
    y = state["kane"]["y"]
    for nx, ny in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
        if can_build(state, nx, ny):
            return True
    return False


def place_building_near_kane(state: dict[str, Any], building_type: str) -> bool:
    x = state["kane"]["x"]
    y = state["kane"]["y"]

    candidates = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]
    random.shuffle(candidates)

    for nx, ny in candidates:
        if can_build(state, nx, ny):
            state["buildings"].append({
                "x": nx,
                "y": ny,
                "type": building_type,
                "tick": state["tick"],
            })
            return True
    return False


def evolve_terrain(state: dict[str, Any]) -> None:
    world = state["world"]
    rows = [list(r) for r in world["tiles"]]

    for b in state["buildings"]:
        x = b["x"]
        y = b["y"]
        if b["type"] == "garden":
            for nx, ny in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
                if 0 < nx < world["width"] - 1 and 0 < ny < world["height"] - 1 and rows[ny][nx] == ".":
                    if random.random() < 0.08:
                        rows[ny][nx] = "^"
        if b["type"] == "ruin":
            if random.random() < 0.03 and 0 < x < world["width"] - 1 and 0 < y < world["height"] - 1:
                rows[y][x] = "*"

    world["tiles"] = ["".join(r) for r in rows]


def apply_action(state: dict[str, Any], decision: dict[str, str]) -> None:
    kane = state["kane"]
    x = kane["x"]
    y = kane["y"]
    action = decision["action"]

    moved = False
    built = False
    built_label = ""

    dx = 0
    dy = 0
    if action == "MOVE_NORTH":
        dy = -1
    elif action == "MOVE_SOUTH":
        dy = 1
    elif action == "MOVE_WEST":
        dx = -1
    elif action == "MOVE_EAST":
        dx = 1

    if action.startswith("MOVE_"):
        nx = x + dx
        ny = y + dy
        if can_walk(state, nx, ny):
            kane["x"] = nx
            kane["y"] = ny
            kane["energy"] = max(0, kane["energy"] - 1)
            moved = True

    if action == "WAIT":
        kane["energy"] = min(100, kane["energy"] + 1)

    build_map = {
        "BUILD_HUT": "hut",
        "BUILD_TOWER": "tower",
        "BUILD_FIRE": "fire",
        "BUILD_RUIN": "ruin",
        "BUILD_GARDEN": "garden",
    }

    if action in build_map:
        built = place_building_near_kane(state, build_map[action])
        if built:
            built_label = BUILDING_TYPES[build_map[action]]["label"]
            kane["energy"] = max(0, kane["energy"] - 2)

    kane["mood"] = decision.get("mood", kane["mood"])
    kane["last_action"] = action
    kane["last_words"] = decision.get("say", kane["last_words"])

    state["tick"] += 1
    state["time_of_day"] = "night" if (state["tick"] // 8) % 2 == 0 else "dawn"

    evolve_terrain(state)

    if built:
        state["log"].append(f"Kane построил: {built_label}. «{kane['last_words']}»")
    elif action.startswith("MOVE_") and moved:
        state["log"].append(f"Kane moved to ({kane['x']}, {kane['y']}). «{kane['last_words']}»")
    elif action.startswith("MOVE_") and not moved:
        state["log"].append(f"Kane упёрся в край мира. «{kane['last_words']}»")
    else:
        state["log"].append(f"Kane ждёт. «{kane['last_words']}»")

    if len(state["log"]) > 120:
        state["log"] = state["log"][-120:]


def render_html() -> str:
    return """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kane Colony</title>
<style>
body{margin:0;background:#0d1117;color:#e6edf3;font-family:Arial,sans-serif}
.wrap{padding:16px}
h1{margin:0 0 8px 0}
.meta{color:#9aa4b2;margin-bottom:14px}
.layout{display:grid;grid-template-columns:minmax(320px,max-content) 1fr;gap:16px}
.panel{background:#11161d;border:1px solid #1f2937;border-radius:14px;padding:12px}
.grid{display:grid;gap:1px;background:#1b2430;width:max-content;padding:6px;border-radius:12px}
.cell{width:20px;height:20px;display:flex;align-items:center;justify-content:center;font-size:11px;color:rgba(255,255,255,.85)}
.log{height:480px;overflow:auto;white-space:pre-wrap;line-height:1.45;font-size:14px}
.badges{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
.badge{background:#1a2230;border:1px solid #2a3648;border-radius:999px;padding:6px 10px;font-size:13px}
.small{color:#9aa4b2}
@media (max-width:900px){.layout{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
<h1>Kane Colony</h1>
<div class="meta">Kane сам строит мир, а состояние сохраняется в файл.</div>
<div class="layout">
  <div class="panel">
    <div id="worldName"></div>
    <div id="worldDesc" class="small" style="margin:8px 0 12px 0;"></div>
    <div id="grid" class="grid"></div>
    <div class="badges">
      <div class="badge">~ вода</div>
      <div class="badge">. трава</div>
      <div class="badge">^ лес</div>
      <div class="badge">* камень</div>
      <div class="badge">K Kane</div>
      <div class="badge">H хижина</div>
      <div class="badge">T башня</div>
      <div class="badge">F костёр</div>
      <div class="badge">R руины</div>
      <div class="badge">G сад</div>
    </div>
  </div>
  <div class="panel">
    <div id="kaneBox" style="line-height:1.5;margin-bottom:12px"></div>
    <div class="log" id="logBox"></div>
  </div>
</div>
</div>

<script>
const colorMap = {"~":"#2b6cb0",".":"#5aa469","^":"#276749","*":"#9aa0a6"};
const buildingMap = {
  hut:{symbol:"H",color:"#8b5e3c"},
  tower:{symbol:"T",color:"#c0c0c0"},
  fire:{symbol:"F",color:"#ff6b35"},
  ruin:{symbol:"R",color:"#6b7280"},
  garden:{symbol:"G",color:"#2ecc71"}
};

function esc(s){
  return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}

function render(state){
  const world = state.world;
  const kane = state.kane;
  const grid = document.getElementById("grid");
  grid.style.gridTemplateColumns = `repeat(${world.width}, 20px)`;

  const buildings = {};
  for(const b of state.buildings){
    buildings[`${b.x},${b.y}`] = b;
  }

  const cells = [];
  for(let y=0;y<world.height;y++){
    const row = world.tiles[y];
    for(let x=0;x<world.width;x++){
      let ch = row[x];
      let bg = colorMap[ch] || "#000";
      let display = ch;

      const key = `${x},${y}`;
      if(buildings[key]){
        const b = buildings[key];
        bg = buildingMap[b.type].color;
        display = buildingMap[b.type].symbol;
      }

      if(x === kane.x && y === kane.y){
        bg = "#c53030";
        display = "K";
      }

      cells.push(`<div class="cell" style="background:${bg}">${display}</div>`);
    }
  }
  grid.innerHTML = cells.join("");

  document.getElementById("worldName").innerHTML = "<b>" + esc(world.name) + "</b>";
  document.getElementById("worldDesc").textContent = world.description;

  document.getElementById("kaneBox").innerHTML =
    "<b>Kane</b><br>" +
    "позиция: (" + kane.x + ", " + kane.y + ")<br>" +
    "настроение: " + esc(kane.mood) + "<br>" +
    "энергия: " + kane.energy + "<br>" +
    "время: " + esc(state.time_of_day) + "<br>" +
    "тик: " + state.tick + "<br>" +
    "построек: " + state.buildings.length + "<br>" +
    "последнее действие: " + esc(kane.last_action) + "<br>" +
    "фраза: “" + esc(kane.last_words) + "”";

  document.getElementById("logBox").innerHTML =
    state.log.slice().reverse().map(x => "• " + esc(x)).join("<br>");
}

async function refresh(){
  try{
    const res = await fetch("/state.json?ts=" + Date.now(), {cache:"no-store"});
    const state = await res.json();
    render(state);
  }catch(e){}
}

refresh();
setInterval(refresh, 500);
</script>
</body>
</html>
"""


class KaneHandler(BaseHTTPRequestHandler):
    state_ref: dict[str, Any] | None = None
    lock_ref: threading.Lock | None = None

    def do_GET(self) -> None:
        if self.path.startswith("/state.json"):
            assert self.state_ref is not None
            assert self.lock_ref is not None
            with self.lock_ref:
                body = json.dumps(self.state_ref, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/" or self.path.startswith("/index.html"):
            body = render_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def simulation_loop(cfg: KaneConfig, state: dict[str, Any], lock: threading.Lock) -> None:
    while True:
        with lock:
            snapshot = json.loads(json.dumps(state, ensure_ascii=False))

        decision = fallback_decision(snapshot)
        if snapshot["tick"] % 5 == 0:
            try:
                decision = ask_kane(cfg, snapshot)
            except Exception:
                pass

        with lock:
            apply_action(state, decision)
            save_state(state)

        time.sleep(cfg.tick_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kane colony")
    parser.add_argument("prompt", nargs="*", help="world prompt")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tick-seconds", type=float, default=2.0)
    parser.add_argument("--reset", action="store_true", help="start from empty world again")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.reset and STATE_PATH.exists():
        STATE_PATH.unlink()

    cfg = KaneConfig(
        model=args.model,
        endpoint=args.endpoint,
        temperature=args.temperature,
        port=args.port,
        tick_seconds=args.tick_seconds,
        prompt=" ".join(args.prompt).strip() or "пустой мрачный мир, который Kane должен сам развить",
    )

    state = initial_state(cfg)
    lock = threading.Lock()

    KaneHandler.state_ref = state
    KaneHandler.lock_ref = lock

    sim_thread = threading.Thread(
        target=simulation_loop,
        args=(cfg, state, lock),
        daemon=True,
    )
    sim_thread.start()

    server = ThreadingHTTPServer(("0.0.0.0", cfg.port), KaneHandler)

    print(f"Kane Colony running on http://0.0.0.0:{cfg.port}/")
    print(f"State file: {STATE_PATH}")
    print(f"Open: http://YOUR_IP:{cfg.port}/")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
