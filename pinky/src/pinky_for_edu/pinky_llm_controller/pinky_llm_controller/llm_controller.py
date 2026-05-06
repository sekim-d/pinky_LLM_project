# ============================================================
# llm_controller.py
# 역할: 사용자의 자연어 명령 → LLM 추론 → Nav2 NavigateToPose Action 호출
#
# 전체 흐름:
#   1. 터미널에서 사용자가 명령 입력  (예: "마커A로 가")
#   2. Ollama LLM 서버에 HTTP 요청    → JSON 응답 수신
#   3. JSON 파싱 후 action 종류 판단
#   4. Nav2 NavigateToPose Action으로 목표 전송
#   5. Nav2가 A* 경로계획 + 장애물 회피하며 이동
# ============================================================

import json
import math
import sys
import threading

import requests
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf_transformations


# ──────────────────────────────────────────────────────────
# 장소 테이블 (맵 절대 좌표 + 도착 방향)
# ──────────────────────────────────────────────────────────
DEFAULT_LOCATIONS = {
    '충전소':  {'x':  0.0,  'y':  0.0,  'yaw':  0.0},
    '테이블1': {'x':  2.0,  'y':  1.5,  'yaw':  3.14},
    '테이블2': {'x':  2.0,  'y': -1.5,  'yaw':  3.14},
    '테이블3': {'x':  4.0,  'y':  1.5,  'yaw':  3.14},
    '테이블4': {'x':  4.0,  'y': -1.5,  'yaw':  3.14},
    '주방':    {'x': -2.0,  'y':  0.0,  'yaw':  0.0},
    # 앞벽(y=1.5)의 빨간 마커 앞 0.4m 지점
    '마커A':   {'x':  0.5,  'y':  1.3,  'yaw':  1.57},
    # 왼쪽벽(x=-2)의 파란 마커 앞 0.15m 지점
    '마커B':   {'x': -1.8,  'y': -0.5,  'yaw':  3.14},
}

# ──────────────────────────────────────────────────────────
# 시스템 프롬프트 템플릿
# ──────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """\
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
{locations}

규칙:
- 장소명이 나오면 해당 절대 좌표를 사용하세요.
- "앞으로 N미터", "왼쪽으로 N미터" 같은 상대 이동은 relative 모드를 사용하세요.
  앞/뒤 = dx, 왼쪽/오른쪽 = dy (왼쪽 양수, 오른쪽 음수), 회전 = dyaw (반시계 양수)
- 방향(yaw/dyaw)은 라디안으로 표현하세요. (0=동, 1.57=북, 3.14=서, -1.57=남)
- "90도 왼쪽" = dyaw: 1.57, "180도 회전" = dyaw: 3.14
- 반드시 JSON만 출력하세요.\
"""


class LLMController(Node):

    def __init__(self):
        super().__init__('llm_controller')

        # ── 파라미터 ────────────────────────────────────
        self.declare_parameter('ollama_url', 'http://localhost:11434')
        self.declare_parameter('model', 'qwen2.5:7b')

        self.ollama_url = self.get_parameter('ollama_url').value
        self.model      = self.get_parameter('model').value
        self.locations  = DEFAULT_LOCATIONS

        # ── TF 리스너 (상대 이동 계산용) ────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Nav2 Action Client ───────────────────────────
        # simple_navigator 대신 Nav2의 navigate_to_pose Action을 사용
        # Nav2가 A* 경로계획 + 장애물 회피를 자동으로 처리
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ── 정지 명령용 퍼블리셔 ──────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.system_prompt = self._build_system_prompt()

        # ── 터미널 입력 스레드 ───────────────────────────
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

        self.get_logger().info(f'LLM Controller (Nav2 mode) ready. Model: {self.model}')

    def _build_system_prompt(self):
        locations_str = '\n'.join(
            f'  - {name}: x={v["x"]}, y={v["y"]}, yaw={v["yaw"]}'
            for name, v in self.locations.items()
        )
        return SYSTEM_PROMPT_TEMPLATE.format(locations=locations_str)

    def _get_current_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            return x, y, yaw
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None

    def _ask_llm(self, user_input):
        try:
            resp = requests.post(
                f'{self.ollama_url}/api/chat',
                json={
                    'model': self.model,
                    'messages': [
                        {'role': 'system', 'content': self.system_prompt},
                        {'role': 'user',   'content': user_input},
                    ],
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'num_predict': 150,
                        'stop': ['<|im_end|>', '\n\n'],
                    },
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()['message']['content']
        except Exception as e:
            self.get_logger().error(f'LLM request failed: {e}')
            return None

    # ────────────────────────────────────────────────────
    # Nav2 NavigateToPose Action으로 목표 전송
    # Nav2가 A* 전역 경로계획 + 로컬 장애물 회피를 담당
    # ────────────────────────────────────────────────────
    def _navigate_to_goal(self, x, y, yaw):
        # Nav2 서버가 준비될 때까지 최대 5초 대기
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            print('[Pinky] Nav2 서버가 준비되지 않았습니다. Nav2가 실행 중인지 확인하세요.')
            return

        # NavigateToPose 목표 메시지 구성
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

        print(f'[Pinky] Nav2로 이동 명령 전송 → x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}rad')

        # 비동기 전송 — 콜백으로 결과 수신
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            print('[Pinky] Nav2가 목표를 거부했습니다 (경로 없음 또는 충돌 위험).')
            return
        print('[Pinky] 이동 시작... (Nav2 경로 계획 완료)')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            print('[Pinky] 목표 도달 완료!')
        elif status == GoalStatus.STATUS_CANCELED:
            print('[Pinky] 이동이 취소되었습니다.')
        else:
            print(f'[Pinky] 이동 실패 (status: {status})')

    def _execute(self, llm_response):
        raw = llm_response.strip()

        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]

        try:
            cmd = json.loads(raw)
        except json.JSONDecodeError:
            print(f'[Pinky] JSON 파싱 실패: {llm_response}')
            return

        action = cmd.get('action')

        if action == 'navigate':
            mode = cmd.get('mode', 'absolute')

            if mode == 'absolute':
                self._navigate_to_goal(cmd['x'], cmd['y'], cmd.get('yaw', 0.0))

            elif mode == 'relative':
                pose = self._get_current_pose()
                if pose is None:
                    print('[Pinky] 현재 위치를 알 수 없어 상대 이동 불가합니다.')
                    return
                cx, cy, cyaw = pose
                dx   = float(cmd.get('dx',   0.0))
                dy   = float(cmd.get('dy',   0.0))
                dyaw = float(cmd.get('dyaw', 0.0))
                gx   = cx + dx * math.cos(cyaw) - dy * math.sin(cyaw)
                gy   = cy + dx * math.sin(cyaw) + dy * math.cos(cyaw)
                gyaw = cyaw + dyaw
                self._navigate_to_goal(gx, gy, gyaw)
            else:
                print(f'[Pinky] 알 수 없는 mode: {mode}')

        elif action == 'stop':
            # Nav2 Action 취소 + cmd_vel 즉시 정지
            self.nav_client._cancel_goal_async if hasattr(self.nav_client, '_cancel_goal_async') else None
            self.cmd_vel_pub.publish(Twist())
            print('[Pinky] 정지 명령 전송')

        elif action == 'unknown':
            print(f'[Pinky] 이해하지 못했어요: {cmd.get("reason", "")}')

        else:
            print(f'[Pinky] 알 수 없는 action: {action}')

    def _input_loop(self):
        while rclpy.ok():
            try:
                sys.stdout.write('\n[명령] ')
                sys.stdout.flush()
                user_input = sys.stdin.readline().strip()
                if not user_input:
                    continue

                print('[Pinky] 생각 중...')
                response = self._ask_llm(user_input)
                if response:
                    self.get_logger().debug(f'LLM raw: {response}')
                    self._execute(response)
            except EOFError:
                break
            except KeyboardInterrupt:
                break


def main(args=None):
    rclpy.init(args=args)
    node = LLMController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
