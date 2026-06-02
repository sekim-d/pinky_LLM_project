import json
import os
import re
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

OLLAMA_URL        = os.getenv("OLLAMA_URL", "http://localhost:11434")
TASK_MANAGER_URL  = os.getenv("TASK_MANAGER_URL", "http://localhost:8090")
MODEL             = os.getenv("MODEL", "qwen2.5:3b")  # 파인튜닝 후 pinky-warehouse로 변경

ROBOT_URLS = {
    "pinky1": "http://192.168.1.81:8001",
    "pinky2": "http://192.168.1.81:8002",
}

SYSTEM_PROMPT = """\
당신은 물류창고 로봇 제어 AI입니다.
사용자의 자연어 명령을 해석하여 반드시 JSON만 출력하세요.
설명 텍스트, 마크다운, 줄바꿈 없이 JSON 한 줄만 출력하세요.

[입고 작업] 렉 번호(1~16)를 지정해 작업 시작:
{"action": "inbound_task", "parameters": {"storage_zone": "zone_<N>"}}

[단순 이동] 로봇 직접 이동 명령:
{"action": "navigate", "parameters": {"location": "<장소키>"}}

[정지]: {"action": "navigate", "parameters": {"location": "stop"}}
[이해불가]: {"action": "unknown", "parameters": {"reason": "<이유>"}}

장소 키: home, loading_zone, unloading_zone, charging, warehouse,
         marker_0~marker_4, stop

렉 번호 변환 규칙 (반드시 따를 것):
  "N번 렉" / "N번 랙" / "N번 구역" / "N번에 넣어줘" → storage_zone: "zone_N"
  예) "3번 렉에 넣어줘" → {"action": "inbound_task", "parameters": {"storage_zone": "zone_3"}}
  예) "7번 랙으로"     → {"action": "inbound_task", "parameters": {"storage_zone": "zone_7"}}
  렉 번호 범위: 1 ~ 16

"멈춰" / "정지" → navigate, location은 stop

JSON만 출력하세요."""


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


app = FastAPI()


class CommandReq(BaseModel):
    text:  str
    robot: str = "pinky1"  # "pinky1" | "pinky2"


def forward_to_task_manager(cmd: dict) -> str:
    zone = cmd.get("parameters", {}).get("storage_zone")
    if not zone:
        return "storage_zone 누락"
    try:
        resp = requests.post(
            f"{TASK_MANAGER_URL}/task",
            json={"storage_zone": zone},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return f"태스크 생성 완료 — task_id: {data.get('task_id')} / 상태: {data.get('state')}"
    except requests.exceptions.ConnectionError:
        return f"태스크매니저 연결 실패 ({TASK_MANAGER_URL})"
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        if status == 409:
            return f"{zone} 이미 점유 중 — 다른 렉을 선택해주세요"
        if status == 503:
            return "jetcobot1 사용 중 — 잠시 후 다시 시도해주세요"
        return f"태스크매니저 오류 ({status}): {e.response.text}"
    except Exception as e:
        return f"태스크매니저 오류: {e}"


@app.post("/command")
async def command(req: CommandReq):
    try:
        raw = ask_llm(req.text)

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("JSON 없음", raw, 0)
        cmd = json.loads(match.group())

        # 입고 작업 → 태스크매니저로 전달
        if cmd.get("action") == "inbound_task":
            result = forward_to_task_manager(cmd)
        else:
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
