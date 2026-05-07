import json
import math

import requests

from config import OLLAMA_URL, MODEL, SYSTEM_PROMPT
from logger import llm_log


# 사용자의 자연어 명령을 Ollama LLM 서버에 전달하고 응답을 받는 함수.
# 동작 순서:
#   1. SYSTEM_PROMPT(장소 좌표 목록 포함)와 사용자 입력을 함께 Ollama에 HTTP POST
#   2. LLM이 명령을 해석해서 JSON 형식 문자열로 응답
#   3. 응답 문자열을 그대로 반환 (파싱은 parse_llm_response에서 따로 함)
# 주요 옵션:
#   - temperature 0.1: 낮을수록 일정한 응답 (JSON 형식 유지에 유리)
#   - num_predict 150: 응답 토큰 수 제한 (JSON만 받으면 되므로 짧게)
#   - stream False: 응답을 한번에 받음
# 인자: text — 사용자 입력 문자열 ("마커A로 가" 등)
# 반환값: LLM 응답 문자열 (JSON 또는 ```json...``` 형태)
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


# LLM 응답 문자열에서 JSON을 파싱해서 딕셔너리로 변환하는 함수.
# LLM이 항상 순수 JSON만 주지 않고 아래처럼 코드블록으로 감쌀 때가 있어서
# 코드블록을 제거하는 전처리를 먼저 한다.
#   ```json
#   {"action": "navigate", "x": 0.5}
#   ```
# 인자: raw — LLM 응답 문자열
# 반환값: 파싱된 명령 딕셔너리 {"action": ..., "x": ..., ...}
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


# 파싱된 JSON 명령을 읽어서 실제 RobotNode 동작으로 연결하는 함수.
# 처리하는 action 종류:
#   - navigate / absolute: LLM이 준 절대 좌표로 Nav2 이동 명령 전송
#   - navigate / relative: 현재 위치 기준 상대 이동 (앞으로 1m 등)
#       현재 로봇 위치(cx, cy, cyaw)를 읽어서 로봇 방향 기준으로 목표 좌표를 계산.
#       수식: gx = cx + dx*cos(cyaw) - dy*sin(cyaw)
#             gy = cy + dx*sin(cyaw) + dy*cos(cyaw)
#       이렇게 하는 이유: 로봇이 어느 방향을 보든 "앞으로"가 실제 맵에서
#       올바른 방향으로 변환되게 하기 위해서.
#   - stop: /cmd_vel에 속도 0을 발행해서 즉시 정지
#   - unknown: LLM이 이해 못한 경우 이유 문자열 반환
# 인자:
#   cmd      — parse_llm_response()가 반환한 딕셔너리
#   ros_node — RobotNode 인스턴스 (navigate_to_goal, stop 등 호출용)
# 반환값: 처리 결과 메시지 문자열
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
