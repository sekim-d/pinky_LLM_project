import math
import numpy as np

# ── LLM ──────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
MODEL      = "qwen2.5:7b"

# ── 장소 좌표 ─────────────────────────────────
DEFAULT_LOCATIONS = {
    '충전소':  {'x':  0.0,  'y':  0.0,  'yaw':  0.0},
    '테이블1': {'x':  2.0,  'y':  1.5,  'yaw':  math.pi},
    '테이블2': {'x':  2.0,  'y': -1.5,  'yaw':  math.pi},
    '테이블3': {'x':  4.0,  'y':  1.5,  'yaw':  math.pi},
    '테이블4': {'x':  4.0,  'y': -1.5,  'yaw':  math.pi},
    '주방':    {'x': -2.0,  'y':  0.0,  'yaw':  0.0},
    '마커A':   {'x':  0.5,  'y':  1.3,  'yaw':  math.pi / 2},
    '마커B':   {'x': -1.556, 'y': -0.736, 'yaw': 2.362},
}


# LLM에게 전달할 시스템 프롬프트를 생성하는 함수.
# DEFAULT_LOCATIONS 딕셔너리를 읽어서 장소 목록을 텍스트로 자동 변환한다.
# 이 함수를 쓰는 이유: 좌표를 DEFAULT_LOCATIONS 한 곳에서만 관리하기 위해서.
# 만약 좌표를 직접 문자열로 적으면, 수정할 때 두 곳을 동시에 바꿔야 해서
# 한 곳을 빠뜨리는 버그가 생길 수 있다.
# 반환값: LLM 시스템 프롬프트 문자열
def _build_system_prompt() -> str:
    location_lines = "\n".join(
        f"  - {name}: x={loc['x']}, y={loc['y']}, yaw={round(loc['yaw'], 3)}"
        for name, loc in DEFAULT_LOCATIONS.items()
    )
    return f"""\
당신은 자율주행 로봇 Pinky의 제어 AI입니다.
사용자의 자연어 명령을 해석하여 반드시 아래 JSON 형식 중 하나로만 응답하세요.
JSON 외의 텍스트는 절대 포함하지 마세요.

[절대 좌표 이동]
{{"action": "navigate", "mode": "absolute", "x": <float>, "y": <float>, "yaw": <float>}}

[현재 위치 기준 상대 이동]
{{"action": "navigate", "mode": "relative", "dx": <float>, "dy": <float>, "dyaw": <float>}}

[정지]
{{"action": "stop"}}

[이해 불가]
{{"action": "unknown", "reason": "<이유>"}}

사용 가능한 장소 목록 (절대 좌표):
{location_lines}

규칙:
- 장소명이 나오면 해당 절대 좌표를 사용하세요.
- 앞/뒤 = dx, 왼쪽/오른쪽 = dy (왼쪽 양수, 오른쪽 음수), 회전 = dyaw
- 거리 단위는 미터, 기본값 1.0m
- 반드시 JSON만 출력하세요."""


# 카메라 내부 파라미터 행렬(intrinsic matrix)을 생성하는 함수.
# camera.xacro 파일에 정의된 카메라 스펙을 기반으로 계산한다.
#   - 해상도: 1280 x 720
#   - 수평 FOV: 1.08 rad (약 61.9도)
# fx, fy: 초점거리 (픽셀 단위). 이 값으로 픽셀 좌표 → 실제 거리 변환이 가능.
# cx, cy: 이미지 중심점 (주점).
# 이 행렬은 ArUco solvePnP에서 마커까지의 실제 3D 거리를 계산할 때 사용된다.
# 반환값: 3x3 카메라 행렬 (numpy float64)
def _build_camera_matrix() -> np.ndarray:
    _CAMERA_WIDTH  = 1280
    _CAMERA_HEIGHT = 720
    _CAMERA_FOV    = 1.08

    fx = (_CAMERA_WIDTH / 2) / math.tan(_CAMERA_FOV / 2)
    cx = _CAMERA_WIDTH  / 2
    cy = _CAMERA_HEIGHT / 2
    return np.array([[fx,  0.0,  cx],
                     [0.0,  fx,  cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


# ── LLM 시스템 프롬프트 ───────────────────────
SYSTEM_PROMPT = _build_system_prompt()

# ── ArUco 카메라 파라미터 ─────────────────────
CAMERA_MATRIX  = _build_camera_matrix()
DIST_COEFFS    = np.zeros((4, 1), dtype=np.float64)  # 시뮬레이터는 렌즈 왜곡 없음
MARKER_SIZE    = 0.3  # 마커 실제 크기 (미터) — world 파일 box size 기준

# 마커 4개 꼭짓점의 실제 3D 좌표 (마커 중심이 원점)
# solvePnP가 이 좌표와 픽셀 좌표를 비교해서 카메라까지 거리를 계산한다
MARKER_OBJ_PTS = np.array([
    [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],   # 왼쪽 위
    [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],   # 오른쪽 위
    [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],   # 오른쪽 아래
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],   # 왼쪽 아래
], dtype=np.float32)
