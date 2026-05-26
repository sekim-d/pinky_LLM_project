import json
import os
import re
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL      = "qwen2.5:3b"

ROBOT_URLS = {
    "pinky1": "http://192.168.1.81:8001",
    "pinky2": "http://192.168.1.81:8002",
}

# settings.py LOCATIONS 키와 일치해야 함
SYSTEM_PROMPT = """\
당신은 자율주행 로봇 Pinky의 제어 AI입니다.
사용자의 자연어 명령을 해석하여 반드시 JSON만 출력하세요.
설명 텍스트, 마크다운, 줄바꿈 없이 JSON 한 줄만 출력하세요.

이동: {"action": "navigate", "parameters": {"location": "<장소키>"}}
정지: {"action": "navigate", "parameters": {"location": "stop"}}
재개: {"action": "resume", "parameters": {}}
운반: {"action": "transport", "parameters": {"pickup": "<출발지>", "delivery": "<목적지>"}}
이해불가: {"action": "unknown", "parameters": {"reason": "<이유>"}}

사용 가능한 장소 키 (반드시 이 키 그대로 사용):
  home(충전소/홈), loading_zone(적재구역/주방), unloading_zone(하역구역),
  zone_A(A구역/테이블A), zone_B(B구역/테이블B), zone_C(C구역/테이블C),
  charging(충전), warehouse(창고/대기),
  marker_0(마커0), marker_1(마커1), marker_2(마커2), marker_3(마커3), marker_4(마커4),
  stop(멈춰/정지/서/스톱)

"마커 N으로 가" / "마커 N" → navigate 액션, location은 marker_N
"멈춰" / "정지" / "서" / "스톱" → navigate 액션, location은 stop

JSON만 출력하세요."""


# ──────────────────────────────────────────────
# LLM 호출
# ──────────────────────────────────────────────
def ask_llm(text: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 50},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def forward_to_robot(robot: str, cmd: dict) -> str:
    base_url = ROBOT_URLS.get(robot)
    if base_url is None:
        return f"알 수 없는 로봇: {robot}. 사용 가능: {list(ROBOT_URLS.keys())}"

    if cmd.get("action") == "unknown":
        return f"이해 불가: {cmd.get('parameters', {}).get('reason', '')}"

    try:
        resp = requests.post(
            f"{base_url}/command",
            json={"action": cmd["action"], "parameters": cmd.get("parameters", {})},
            timeout=5,
        )
        resp.raise_for_status()
        return f"{robot} → {resp.json()}"
    except requests.exceptions.ConnectionError:
        return f"{robot} 연결 실패 ({base_url})"
    except requests.exceptions.Timeout:
        return f"{robot} 응답 없음 (timeout)"
    except Exception as e:
        return f"{robot} 오류: {e}"


# ──────────────────────────────────────────────
# FastAPI
# ──────────────────────────────────────────────
app = FastAPI()


class CommandReq(BaseModel):
    text:  str
    robot: str = "pinky1"  # "pinky1" | "pinky2"


@app.post("/command")
async def command(req: CommandReq):
    try:
        raw = ask_llm(req.text)

        # 텍스트 앞뒤 잡동사니 제거 — {...} 블록만 추출
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("JSON 없음", raw, 0)
        cmd = json.loads(match.group())
        result = forward_to_robot(req.robot, cmd)
        return {"ok": True, "llm_json": cmd, "result": result}

    except json.JSONDecodeError:
        return JSONResponse(status_code=400,
                            content={"ok": False, "error": f"JSON 파싱 실패: {raw}",
                                     "tip": "LLM이 JSON 외 텍스트를 포함했습니다."})
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"ok": False, "error": str(e)})


@app.get("/status/{robot}")
async def status(robot: str):
    base_url = ROBOT_URLS.get(robot)
    if base_url is None:
        return JSONResponse(status_code=404,
                            content={"ok": False, "error": f"알 수 없는 로봇: {robot}"})
    try:
        resp = requests.get(f"{base_url}/status", timeout=3)
        return resp.json()
    except Exception as e:
        return JSONResponse(status_code=503, content={"ok": False, "error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(Path(__file__).parent / "index.html", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
