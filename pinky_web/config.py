import math
import numpy as np

# ── LLM ──────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
MODEL      = "qwen2.5:7b"

# ── 장소 좌표 ─────────────────────────────────
DEFAULT_LOCATIONS = {
    '충전소':  {'x':  0.0,  'y':  0.0,  'yaw':  0.0},
    '테이블1': {'x':  2.0,  'y':  1.5,  'yaw':  3.14},
    '테이블2': {'x':  2.0,  'y': -1.5,  'yaw':  3.14},
    '테이블3': {'x':  4.0,  'y':  1.5,  'yaw':  3.14},
    '테이블4': {'x':  4.0,  'y': -1.5,  'yaw':  3.14},
    '주방':    {'x': -2.0,  'y':  0.0,  'yaw':  0.0},
    '마커A':   {'x':  0.5,  'y':  1.3,  'yaw':  1.57},
    '마커B':   {'x': -1.556, 'y': -0.736, 'yaw': 2.362},
}

# ── LLM 시스템 프롬프트 ───────────────────────
SYSTEM_PROMPT = """\
당신은 자율주행 로봇 Pinky의 제어 AI입니다.
사용자의 자연어 명령을 해석하여 반드시 아래 JSON 형식 중 하나로만 응답하세요.
JSON 외의 텍스트는 절대 포함하지 마세요.

[절대 좌표 이동]
{"action": "navigate", "mode": "absolute", "x": <float>, "y": <float>, "yaw": <float>}

[현재 위치 기준 상대 이동]
{"action": "navigate", "mode": "relative", "dx": <float>, "dy": <float>, "dyaw": <float>}

[정지]
{"action": "stop"}

[이해 불가]
{"action": "unknown", "reason": "<이유>"}

사용 가능한 장소 목록 (절대 좌표):
  - 충전소: x=0.0, y=0.0, yaw=0.0
  - 테이블1: x=2.0, y=1.5, yaw=3.14
  - 테이블2: x=2.0, y=-1.5, yaw=3.14
  - 테이블3: x=4.0, y=1.5, yaw=3.14
  - 테이블4: x=4.0, y=-1.5, yaw=3.14
  - 주방: x=-2.0, y=0.0, yaw=0.0
  - 마커A: x=0.5, y=1.3, yaw=1.57
  - 마커B: x=-1.556, y=-0.736, yaw=2.362

규칙:
- 장소명이 나오면 해당 절대 좌표를 사용하세요.
- 앞/뒤 = dx, 왼쪽/오른쪽 = dy (왼쪽 양수, 오른쪽 음수), 회전 = dyaw
- 거리 단위는 미터, 기본값 1.0m
- 반드시 JSON만 출력하세요."""

# ── ArUco 카메라 파라미터 (camera.xacro 기준) ──
# 해상도: 1280x720, 수평 FOV: 1.08rad
_fx = (1280 / 2) / math.tan(1.08 / 2)
CAMERA_MATRIX = np.array([[_fx,  0.0, 640.0],
                           [0.0, _fx,  360.0],
                           [0.0,  0.0,   1.0]], dtype=np.float64)
DIST_COEFFS  = np.zeros((4, 1), dtype=np.float64)
MARKER_SIZE  = 0.3  # 마커 실제 크기 (미터)

MARKER_OBJ_PTS = np.array([
    [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
], dtype=np.float32)
