import json
import math
import threading

import requests
import rclpy
from action_msgs.msg import GoalStatus
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from pydantic import BaseModel
from rclpy.action import ActionClient
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf_transformations
import uvicorn

OLLAMA_URL = "http://localhost:11434"
MODEL      = "qwen2.5:7b"

DEFAULT_LOCATIONS = {
    '충전소':  {'x':  0.0,  'y':  0.0,  'yaw':  0.0},
    '테이블1': {'x':  2.0,  'y':  1.5,  'yaw':  3.14},
    '테이블2': {'x':  2.0,  'y': -1.5,  'yaw':  3.14},
    '테이블3': {'x':  4.0,  'y':  1.5,  'yaw':  3.14},
    '테이블4': {'x':  4.0,  'y': -1.5,  'yaw':  3.14},
    '주방':    {'x': -2.0,  'y':  0.0,  'yaw':  0.0},
    '마커A':   {'x':  0.5,  'y':  1.3,  'yaw':  1.57},
    '마커B':   {'x': -1.8,  'y': -0.5,  'yaw':  3.14},
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
  - 마커A: x=0.5, y=1.3, yaw=1.57
  - 마커B: x=-1.8, y=-0.5, yaw=3.14

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

        # Nav2 Action Client — simple_navigator 대신 Nav2가 경로계획 + 장애물 회피 담당
        self.nav_client  = ActionClient(self, NavigateToPose, 'navigate_to_pose')
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

    def navigate_to_goal(self, x, y, yaw):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            return False, "Nav2 서버가 준비되지 않았습니다."

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.stamp    = self.get_clock().now().to_msg()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, float(yaw))
        goal_msg.pose.pose.orientation.x = q[0]
        goal_msg.pose.pose.orientation.y = q[1]
        goal_msg.pose.pose.orientation.z = q[2]
        goal_msg.pose.pose.orientation.w = q[3]

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_callback)
        return True, f"Nav2 이동 명령 전송 → x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}rad"

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Nav2가 목표를 거부했습니다.')
            return
        self.get_logger().info('이동 시작...')
        goal_handle.get_result_async().add_done_callback(self._result_callback)

    def _result_callback(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('목표 도달 완료!')
        else:
            self.get_logger().warn(f'이동 실패 (status: {status})')

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
    with open("/home/seongeun/pinky_LLM_project/pinky_web/index.html", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
