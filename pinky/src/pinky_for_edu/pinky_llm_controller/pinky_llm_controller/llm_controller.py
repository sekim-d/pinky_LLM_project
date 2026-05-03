# ============================================================
# llm_controller.py
# 역할: 사용자의 자연어 명령 → LLM 추론 → ROS2 토픽 발행
#
# 전체 흐름:
#   1. 터미널에서 사용자가 명령 입력  (예: "앞으로 가")
#   2. Ollama LLM 서버에 HTTP 요청    → JSON 응답 수신
#   3. JSON 파싱 후 action 종류 판단
#   4. /goal_pose 또는 /cmd_vel 토픽으로 ROS2 메시지 발행
#   5. simple_navigator 가 메시지를 받아 가제보 로봇을 실제로 이동
# ============================================================

import json       # LLM이 반환하는 JSON 문자열을 파이썬 딕셔너리로 변환할 때 사용
import math       # 상대 이동 시 로봇 방향(yaw)을 고려한 좌표 변환에 사용
import sys        # 터미널 입출력 (stdin/stdout) 직접 제어에 사용
import threading  # 터미널 입력 대기와 ROS2 스핀을 동시에 실행하기 위해 사용

import requests   # Ollama LLM 서버에 HTTP POST 요청을 보내는 라이브러리
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
# PoseStamped : 목표 위치(x, y, yaw)를 담는 ROS2 메시지 타입 → /goal_pose 토픽
# Twist       : 선속도·각속도를 담는 ROS2 메시지 타입       → /cmd_vel 토픽 (정지 시)
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
# TF(Transform) : ROS2에서 좌표계 간 변환 정보를 관리하는 시스템
# Buffer/TransformListener : "map 기준으로 로봇이 지금 어디 있는지" 를 가져올 때 사용
import tf_transformations
# yaw(라디안) ↔ 쿼터니언 변환 유틸리티
# ROS2 PoseStamped 는 방향을 쿼터니언으로 표현하므로 변환이 필요


# ──────────────────────────────────────────────────────────
# 장소 테이블
# 장소명 → 맵 절대 좌표 (x, y) + 도착 후 방향 (yaw, 라디안)
# "테이블1로 가" 명령 시 LLM이 이 좌표를 시스템 프롬프트에서 참조해 JSON에 담아줌
# ──────────────────────────────────────────────────────────
DEFAULT_LOCATIONS = {
    '충전소':  {'x':  0.0, 'y':  0.0, 'yaw': 0.0},
    '테이블1': {'x':  2.0, 'y':  1.5, 'yaw': 3.14},
    '테이블2': {'x':  2.0, 'y': -1.5, 'yaw': 3.14},
    '테이블3': {'x':  4.0, 'y':  1.5, 'yaw': 3.14},
    '테이블4': {'x':  4.0, 'y': -1.5, 'yaw': 3.14},
    '주방':    {'x': -2.0, 'y':  0.0, 'yaw': 0.0},
}

# ──────────────────────────────────────────────────────────
# 시스템 프롬프트 템플릿
# LLM에게 "어떤 형식으로 대답해야 하는지" 를 알려주는 지시문
# 이 내용이 API 요청의 system 역할 메시지로 매번 함께 전송됨
#
# {locations} 부분은 실행 시 DEFAULT_LOCATIONS 내용으로 채워짐
# {{...}} 는 파이썬 .format() 에서 중괄호 자체를 출력하기 위한 이스케이프
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


# ──────────────────────────────────────────────────────────
# LLMController 클래스
# ROS2 Node 를 상속받아 만든 핵심 클래스
# Node 를 상속한다는 것 = 이 클래스가 ROS2 네트워크의 참여자(노드)가 된다는 의미
# ──────────────────────────────────────────────────────────
class LLMController(Node):

    def __init__(self):
        # ROS2 네트워크에 'llm_controller' 라는 이름으로 노드 등록
        super().__init__('llm_controller')

        # ── 파라미터 선언 ───────────────────────────────
        # launch 파일이나 커맨드라인에서 값을 주입할 수 있도록 파라미터로 선언
        # 기본값: localhost Ollama 서버, qwen2.5:7b 모델
        self.declare_parameter('ollama_url', 'http://localhost:11434')
        self.declare_parameter('model', 'qwen2.5:7b')

        self.ollama_url = self.get_parameter('ollama_url').value
        self.model      = self.get_parameter('model').value
        self.locations  = DEFAULT_LOCATIONS

        # ── TF 리스너 설정 ──────────────────────────────
        # 가제보 시뮬레이터가 로봇의 실시간 위치를 TF 트리로 발행함
        # Buffer : TF 데이터를 시간순으로 저장하는 캐시
        # TransformListener : /tf, /tf_static 토픽을 구독해 Buffer 에 채워줌
        # → lookup_transform('map', 'base_link') 로 "맵 기준 로봇 위치" 조회 가능
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── ROS2 퍼블리셔 설정 ──────────────────────────
        # /goal_pose : simple_navigator 가 구독 → 목표 지점으로 PID 이동 시작
        # /cmd_vel   : 가제보 브리지가 구독 → 로봇 바퀴 속도 직접 제어 (정지 시 사용)
        self.goal_pub    = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # 장소 목록을 시스템 프롬프트 텍스트에 삽입해 완성
        self.system_prompt = self._build_system_prompt()

        # ── 터미널 입력 스레드 ──────────────────────────
        # input() 은 블로킹(대기) 함수라서 메인 스레드에서 쓰면 ROS2 스핀이 멈춤
        # 별도 데몬 스레드에서 입력을 기다리고, 메인 스레드는 ROS2 이벤트 처리 전담
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

        self.get_logger().info(f'LLM Controller ready. Model: {self.model}')
        self.get_logger().info(f'Ollama: {self.ollama_url}')

    # ────────────────────────────────────────────────────
    # 시스템 프롬프트 완성
    # DEFAULT_LOCATIONS 의 장소 정보를 읽어 텍스트로 변환 후
    # SYSTEM_PROMPT_TEMPLATE 의 {locations} 자리에 삽입
    # ────────────────────────────────────────────────────
    def _build_system_prompt(self):
        locations_str = '\n'.join(
            f'  - {name}: x={v["x"]}, y={v["y"]}, yaw={v["yaw"]}'
            for name, v in self.locations.items()
        )
        return SYSTEM_PROMPT_TEMPLATE.format(locations=locations_str)

    # ────────────────────────────────────────────────────
    # 현재 로봇 위치 조회 (TF 사용)
    # 가제보가 발행하는 TF 트리에서 map → base_link 변환을 읽어
    # 로봇의 현재 (x, y, yaw) 를 반환
    # 실패 시 None 반환 (가제보가 꺼져 있거나 TF 아직 미수신 시)
    # ────────────────────────────────────────────────────
    def _get_current_pose(self):
        try:
            # Time() = "가장 최신 변환 데이터를 달라" 는 의미
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation  # 쿼터니언 형식의 방향값
            # 쿼터니언 → 오일러각 변환 후 yaw(수평 회전각)만 추출
            _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            return x, y, yaw
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None

    # ────────────────────────────────────────────────────
    # LLM 서버에 명령 전송 및 응답 수신
    # Ollama의 /api/chat 엔드포인트에 HTTP POST 요청
    # system 역할: LLM에게 JSON만 출력하라는 지시문
    # user 역할  : 사용자가 실제로 입력한 명령어
    # 반환값     : LLM이 생성한 JSON 문자열 (예: '{"action":"navigate",...}')
    # ────────────────────────────────────────────────────
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
                    'stream': False,   # 스트리밍 없이 전체 응답을 한 번에 받음
                    'options': {
                        'temperature': 0.1,        # 낮을수록 일관된 출력 (0=완전 고정)
                        'num_predict': 150,         # 최대 생성 토큰 수 (JSON은 짧으므로 150으로 제한)
                        'stop': ['<|im_end|>', '\n\n'],  # 이 문자열이 나오면 생성 중단
                    },
                },
                timeout=30,  # 30초 이내 응답 없으면 에러 처리
            )
            resp.raise_for_status()  # HTTP 4xx/5xx 에러 시 예외 발생
            return resp.json()['message']['content']  # LLM 응답 텍스트 반환
        except Exception as e:
            self.get_logger().error(f'LLM request failed: {e}')
            return None

    # ────────────────────────────────────────────────────
    # 목표 좌표를 ROS2 메시지로 변환 후 /goal_pose 토픽으로 발행
    # simple_navigator 가 이 메시지를 받으면 PID 제어로 이동 시작
    #
    # yaw(라디안) → 쿼터니언 변환이 필요한 이유:
    #   ROS2 PoseStamped 의 orientation 필드는 쿼터니언만 허용
    #   (쿼터니언은 3D 회전을 4개 숫자로 표현하는 방식)
    # ────────────────────────────────────────────────────
    def _publish_goal(self, x, y, yaw):
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()  # 현재 시각 (시뮬 시간)
        msg.header.frame_id = 'map'          # 좌표계: 맵 기준
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0            # 지면 위 이동이므로 z=0 고정
        # roll=0, pitch=0, yaw 만 변환 (평면 이동 로봇은 수평 회전만 필요)
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, float(yaw))
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        self.goal_pub.publish(msg)
        print(f'[Pinky] 목표 좌표 전송 → x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}rad')

    # ────────────────────────────────────────────────────
    # LLM 응답(JSON 문자열)을 해석해 실제 ROS2 동작으로 실행
    #
    # action 종류:
    #   navigate / absolute : 맵의 절대 좌표로 이동
    #   navigate / relative : 현재 위치에서 상대적으로 이동
    #   stop                : 즉시 정지 (속도 0 전송)
    #   unknown             : LLM이 명령을 이해 못함
    # ────────────────────────────────────────────────────
    def _execute(self, llm_response):
        raw = llm_response.strip()

        # LLM이 가끔 ```json ... ``` 형태로 감싸서 응답하므로 제거
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]

        # 문자열 → 파이썬 딕셔너리 변환
        try:
            cmd = json.loads(raw)
        except json.JSONDecodeError:
            print(f'[Pinky] JSON 파싱 실패: {llm_response}')
            return

        action = cmd.get('action')

        # ── 이동 명령 ────────────────────────────────
        if action == 'navigate':
            mode = cmd.get('mode', 'absolute')

            if mode == 'absolute':
                # 장소명 명령 시 (예: "테이블1로 가")
                # LLM이 장소 테이블을 참조해 절대 좌표를 JSON에 직접 채워줌
                self._publish_goal(cmd['x'], cmd['y'], cmd.get('yaw', 0.0))

            elif mode == 'relative':
                # 방향 명령 시 (예: "앞으로 1미터")
                # 현재 로봇 위치를 TF에서 읽어온 뒤 상대 이동량을 더해 절대 좌표 계산
                pose = self._get_current_pose()
                if pose is None:
                    print('[Pinky] 현재 위치를 알 수 없어 상대 이동 불가합니다.')
                    return
                cx, cy, cyaw = pose
                dx   = float(cmd.get('dx',   0.0))  # 로봇 기준 앞(+) / 뒤(-) 거리 (미터)
                dy   = float(cmd.get('dy',   0.0))  # 로봇 기준 왼쪽(+) / 오른쪽(-) 거리
                dyaw = float(cmd.get('dyaw', 0.0))  # 회전량 (라디안, 반시계 방향이 양수)

                # 로봇 좌표계의 dx/dy를 맵 좌표계로 회전 변환
                # 로봇이 현재 cyaw 방향을 바라보고 있으므로 그 방향 기준으로 계산
                gx   = cx + dx * math.cos(cyaw) - dy * math.sin(cyaw)
                gy   = cy + dx * math.sin(cyaw) + dy * math.cos(cyaw)
                gyaw = cyaw + dyaw
                self._publish_goal(gx, gy, gyaw)
            else:
                print(f'[Pinky] 알 수 없는 mode: {mode}')

        # ── 정지 명령 ────────────────────────────────
        elif action == 'stop':
            # 빈 Twist 메시지(모든 속도=0)를 /cmd_vel 로 발행 → 즉시 정지
            self.cmd_vel_pub.publish(Twist())
            print('[Pinky] 정지 명령 전송')

        # ── 이해 불가 ────────────────────────────────
        elif action == 'unknown':
            print(f'[Pinky] 이해하지 못했어요: {cmd.get("reason", "")}')

        else:
            print(f'[Pinky] 알 수 없는 action: {action}')

    # ────────────────────────────────────────────────────
    # 터미널 입력 루프 (별도 스레드에서 실행)
    # 사용자 입력을 기다렸다가 LLM에 전달하고 결과를 실행하는 메인 사이클
    # ────────────────────────────────────────────────────
    def _input_loop(self):
        while rclpy.ok():  # ROS2 가 종료되지 않은 동안 반복
            try:
                # sys.stdout.write + flush : ros2 launch 환경에서도 프롬프트가 즉시 표시되도록
                # (일반 print 는 줄바꿈 없으면 버퍼에 쌓여 보이지 않을 수 있음)
                sys.stdout.write('\n[명령] ')
                sys.stdout.flush()
                user_input = sys.stdin.readline().strip()  # 엔터 입력까지 대기
                if not user_input:
                    continue  # 빈 줄이면 다시 대기

                print('[Pinky] 생각 중...')
                response = self._ask_llm(user_input)   # LLM 호출 (최대 30초 대기)
                if response:
                    self.get_logger().debug(f'LLM raw: {response}')
                    self._execute(response)             # JSON 파싱 후 ROS2 동작 실행
            except EOFError:
                break  # stdin 이 닫히면 (파이프 입력 종료 등) 루프 종료
            except KeyboardInterrupt:
                break  # Ctrl+C 로 종료


# ──────────────────────────────────────────────────────────
# 진입점
# ros2 run 또는 ros2 launch 로 실행될 때 main() 이 호출됨
# ──────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)          # ROS2 초기화 (네트워크 연결)
    node = LLMController()         # 노드 생성 (퍼블리셔·TF·스레드 모두 여기서 시작)
    try:
        rclpy.spin(node)           # ROS2 이벤트 루프 시작 (토픽 수신·타이머 처리)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down.')
    finally:
        node.destroy_node()        # 노드 자원 해제
        rclpy.shutdown()           # ROS2 종료


if __name__ == '__main__':
    main()
