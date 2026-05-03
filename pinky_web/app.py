import json
import math
import threading

import requests
import rclpy
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from geometry_msgs.msg import PoseStamped, Twist
from pydantic import BaseModel
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf_transformations
import uvicorn

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
MODEL      = "qwen2.5:7b"

DEFAULT_LOCATIONS = {
    '충전소':  {'x':  0.0, 'y':  0.0, 'yaw': 0.0},
    '테이블1': {'x':  2.0, 'y':  1.5, 'yaw': 3.14},
    '테이블2': {'x':  2.0, 'y': -1.5, 'yaw': 3.14},
    '테이블3': {'x':  4.0, 'y':  1.5, 'yaw': 3.14},
    '테이블4': {'x':  4.0, 'y': -1.5, 'yaw': 3.14},
    '주방':    {'x': -2.0, 'y':  0.0, 'yaw': 0.0},
}

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

규칙:
- 장소명이 나오면 해당 절대 좌표를 사용하세요.
- 앞/뒤 = dx, 왼쪽/오른쪽 = dy (왼쪽 양수, 오른쪽 음수), 회전 = dyaw
- 거리 단위는 미터, 기본값 1.0m
- 반드시 JSON만 출력하세요."""


# ──────────────────────────────────────────────
# ROS2 노드 (백그라운드 스레드)
# ──────────────────────────────────────────────
class WebRosNode(Node):
    def __init__(self):
        super().__init__('web_controller')
        self.goal_pub    = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def get_current_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=3),
            )
            x   = t.transform.translation.x
            y   = t.transform.translation.y
            q   = t.transform.rotation
            _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            return x, y, yaw
        except Exception:
            return None

    def publish_goal(self, x, y, yaw):
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, float(yaw))
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        self.goal_pub.publish(msg)

    def stop(self):
        self.cmd_vel_pub.publish(Twist())


rclpy.init()
ros_node = WebRosNode()
threading.Thread(target=lambda: rclpy.spin(ros_node), daemon=True).start()


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
            "options": {"temperature": 0.1, "num_predict": 150},
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def execute_cmd(cmd: dict) -> str:
    action = cmd.get("action")

    if action == "navigate":
        mode = cmd.get("mode", "absolute")
        if mode == "absolute":
            x, y, yaw = cmd["x"], cmd["y"], cmd.get("yaw", 0.0)
            ros_node.publish_goal(x, y, yaw)
            return f"목표 전송 → x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}"
        elif mode == "relative":
            pose = ros_node.get_current_pose()
            if pose is None:
                return "현재 위치를 알 수 없어 상대 이동 불가 (TF 없음)"
            cx, cy, cyaw = pose
            dx, dy, dyaw = float(cmd.get("dx", 0)), float(cmd.get("dy", 0)), float(cmd.get("dyaw", 0))
            gx   = cx + dx * math.cos(cyaw) - dy * math.sin(cyaw)
            gy   = cy + dx * math.sin(cyaw) + dy * math.cos(cyaw)
            gyaw = cyaw + dyaw
            ros_node.publish_goal(gx, gy, gyaw)
            return f"상대 이동 → dx={dx}, dy={dy}, dyaw={dyaw} (목표: x={gx:.2f}, y={gy:.2f})"

    elif action == "stop":
        ros_node.stop()
        return "정지 명령 전송"

    elif action == "unknown":
        return f"이해 불가: {cmd.get('reason', '')}"

    return f"알 수 없는 action: {action}"


# ──────────────────────────────────────────────
# FastAPI
# ──────────────────────────────────────────────
app = FastAPI()


class CommandReq(BaseModel):
    text: str


@app.post("/command")
async def command(req: CommandReq):
    try:
        raw = ask_llm(req.text)

        # 백틱 코드블록 제거
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]

        cmd    = json.loads(clean)
        result = execute_cmd(cmd)
        return {"ok": True, "llm_json": cmd, "result": result}

    except json.JSONDecodeError:
        return JSONResponse(status_code=400,
                            content={"ok": False, "error": f"JSON 파싱 실패: {raw}"})
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"ok": False, "error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("/home/seongeun/pinky_web/index.html", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
