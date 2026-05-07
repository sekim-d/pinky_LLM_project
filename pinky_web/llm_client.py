import json
import math

import requests

from config import OLLAMA_URL, MODEL, SYSTEM_PROMPT
from logger import llm_log


def ask_llm(text: str) -> str:
    llm_log.info(f"요청 전송 → \"{text}\"  (model: {MODEL})")
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 150},
        },
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]
    llm_log.info(f"응답 수신 ← {raw.strip()}")
    return raw


def parse_llm_response(raw: str) -> dict:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    cmd = json.loads(clean)
    llm_log.info(
        f"파싱 완료 → action={cmd.get('action')}"
        + (f"  mode={cmd.get('mode')}" if "mode" in cmd else "")
        + (f"  x={cmd.get('x')}  y={cmd.get('y')}  yaw={cmd.get('yaw')}"
           if cmd.get("mode") == "absolute" else "")
        + (f"  dx={cmd.get('dx')}  dy={cmd.get('dy')}  dyaw={cmd.get('dyaw')}"
           if cmd.get("mode") == "relative" else "")
    )
    return cmd


def execute_cmd(cmd: dict, ros_node) -> str:
    action = cmd.get("action")

    if action == "navigate":
        mode = cmd.get("mode", "absolute")
        if mode == "absolute":
            x, y, yaw = cmd["x"], cmd["y"], cmd.get("yaw", 0.0)
            _, msg = ros_node.navigate_to_goal(x, y, yaw)
            return msg
        elif mode == "relative":
            pose = ros_node.get_current_pose()
            if pose is None:
                return "현재 위치를 알 수 없어 상대 이동 불가 (TF 없음)"
            cx, cy, cyaw = pose
            dx   = float(cmd.get("dx", 0))
            dy   = float(cmd.get("dy", 0))
            dyaw = float(cmd.get("dyaw", 0))
            gx   = cx + dx * math.cos(cyaw) - dy * math.sin(cyaw)
            gy   = cy + dx * math.sin(cyaw) + dy * math.cos(cyaw)
            gyaw = cyaw + dyaw
            _, msg = ros_node.navigate_to_goal(gx, gy, gyaw)
            return msg

    elif action == "stop":
        ros_node.stop()
        return "정지 명령 전송"

    elif action == "unknown":
        return f"이해 불가: {cmd.get('reason', '')}"

    return f"알 수 없는 action: {action}"
