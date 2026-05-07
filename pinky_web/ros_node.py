import math
import threading

import cv2
import numpy as np
import rclpy
import tf2_geometry_msgs
import tf_transformations
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from config import CAMERA_MATRIX, DIST_COEFFS, MARKER_OBJ_PTS
from logger import nav_log, aruco_log

_aruco_dict    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_aruco_params  = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)


class RobotNode(Node):

    # ROS2 노드 초기화 함수. app.py 시작 시 한 번 호출된다.
    # 등록하는 것들:
    #   - Nav2 ActionClient: navigate_to_pose 액션 서버에 이동 목표 전송용
    #   - cmd_vel Publisher: 긴급 정지 시 속도 0을 직접 발행하는 용도
    #   - TF Buffer/Listener: 좌표계 변환 정보를 실시간으로 수신해서 저장
    #   - Image Subscriber: /camera/image_raw 토픽 구독 → 이미지 올 때마다 _image_cb 호출
    #   - detected_markers: 카메라로 감지된 마커 ID와 map 좌표를 저장하는 딕셔너리
    #   - auto_navigate: True면 마커 감지 시 자동으로 이동 명령 전송
    #   - _last_nav_time: 연속 이동 명령 방지용 타임스탬프 (3초 쿨다운)
    def __init__(self):
        super().__init__('web_controller',
                         parameter_overrides=[
                             Parameter('use_sim_time', Parameter.Type.BOOL, True)
                         ])

        self.nav_client  = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.bridge           = CvBridge()
        self.detected_markers = {}
        self.auto_navigate    = False
        self._last_nav_time   = 0.0

        self.create_subscription(Image, '/camera/image_raw', self._image_cb, 10)
        nav_log.info("RobotNode 초기화 완료 — Nav2 + ArUco 대기 중")

    # 카메라 이미지가 들어올 때마다 자동으로 호출되는 콜백 함수 (30Hz).
    # 처리 순서:
    #   1. CvBridge로 ROS2 Image 메시지 → OpenCV numpy 배열(BGR) 변환
    #   2. BGR → 흑백 변환 (ArUco 감지는 흑백 이미지로 수행)
    #   3. ARUCO_DETECTOR로 마커 감지 → corners(꼭짓점 픽셀 좌표), ids(마커 ID) 추출
    #   4. solvePnP로 픽셀 좌표 → 카메라 기준 3D 좌표(tvec) 계산
    #   5. _camera_to_map()으로 카메라 좌표 → 맵 절대 좌표 변환
    #   6. detected_markers 딕셔너리에 저장
    #   7. auto_navigate=True 이고 3초 쿨다운이 지났으면 마커 앞으로 자동 이동
    # 3초 쿨다운이 필요한 이유:
    #   30Hz로 이미지가 들어오는데 매 프레임마다 이동 명령을 보내면
    #   Nav2가 과부하 걸려서 목표를 계속 거부한다.
    def _image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
        if ids is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        for i, mid in enumerate(ids.flatten()):
            _, _, tvec = cv2.solvePnP(
                MARKER_OBJ_PTS, corners[i][0], CAMERA_MATRIX, DIST_COEFFS)
            tvec = tvec.flatten()
            dist = float(np.linalg.norm(tvec))

            aruco_log.info(
                f"마커 ID={mid} 감지 — "
                f"카메라 기준 ({tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f})m  "
                f"거리={dist:.2f}m"
            )

            world_xy = self._camera_to_map(tvec, msg.header.stamp)
            if world_xy is None:
                aruco_log.warning(f"마커 ID={mid} — TF 변환 실패 (map 좌표 없음)")
                continue

            mx, my = world_xy
            self.detected_markers[int(mid)] = {'x': round(mx, 3), 'y': round(my, 3)}
            aruco_log.info(f"마커 ID={mid} — map 좌표 ({mx:.3f}, {my:.3f})")

            if self.auto_navigate and (now - self._last_nav_time) > 3.0:
                self._last_nav_time = now
                approach_tvec = np.array([tvec[0], tvec[1],
                                          max(tvec[2] - 0.1, 0.05)])
                approach_xy = self._camera_to_map(approach_tvec, msg.header.stamp)
                gx, gy = approach_xy if approach_xy else (mx, my)
                pose = self.get_current_pose()
                yaw  = math.atan2(gy - pose[1], gx - pose[0]) if pose else 0.0
                aruco_log.info(
                    f"자동 이동 시작 — 마커 ID={mid}  "
                    f"목표 ({gx:.2f}, {gy:.2f})  yaw={yaw:.2f}rad"
                )
                self.navigate_to_goal(gx, gy, yaw)

    # 카메라 좌표계 기준 3D 좌표를 맵(map) 기준 절대 좌표로 변환하는 함수.
    # TF 트리를 타고 변환한다:
    #   camera_rgb_frame → camera_link → base_link → odom → map
    # TF가 이 경로의 모든 변환 정보를 실시간으로 가지고 있어서,
    # lookup_transform('map', 'camera_rgb_frame') 한 번으로
    # 카메라에서 맵까지의 전체 변환을 가져올 수 있다.
    # 인자:
    #   tvec  — solvePnP가 계산한 카메라 기준 3D 좌표 [x, y, z] (미터)
    #   stamp — 이미지의 타임스탬프 (TF 시간 동기화용)
    # 반환값: (map_x, map_y) 튜플, 실패 시 None
    def _camera_to_map(self, tvec, stamp):
        try:
            pt = PointStamped()
            pt.header.stamp    = stamp
            pt.header.frame_id = 'camera_rgb_frame'
            pt.point.x = float(tvec[0])
            pt.point.y = float(tvec[1])
            pt.point.z = float(tvec[2])
            tf = self.tf_buffer.lookup_transform(
                'map', 'camera_rgb_frame', rclpy.time.Time())
            out = tf2_geometry_msgs.do_transform_point(pt, tf)
            return out.point.x, out.point.y
        except Exception as e:
            self.get_logger().debug(f'TF 변환 실패: {e}')
            return None

    # TF에서 현재 로봇의 맵 기준 위치와 방향을 읽어오는 함수.
    # map → base_link TF를 조회해서 (x, y, yaw)를 반환한다.
    # ROS2는 방향을 quaternion(x,y,z,w)으로 저장하기 때문에
    # euler_from_quaternion으로 사람이 이해하기 쉬운 yaw 각도로 변환한다.
    # 주로 상대 이동 명령("앞으로 1m")에서 현재 위치를 기준으로
    # 목표 좌표를 계산할 때 사용한다.
    # 반환값: (x, y, yaw) 튜플, TF 없으면 None
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

    # Nav2에 이동 목표를 전송하는 함수.
    # NavigateToPose 액션 메시지를 구성해서 Nav2 bt_navigator에 비동기로 전송한다.
    # 비동기로 보내는 이유: Nav2가 경로를 계획하는 동안 웹서버가 멈추면 안 되기 때문.
    # yaw를 quaternion으로 변환하는 이유:
    #   ROS2는 방향을 항상 quaternion(x,y,z,w) 형식으로 주고받는다.
    #   사람이 쓰는 yaw(라디안) 값을 quaternion_from_euler로 변환해서 넣어야 한다.
    # 콜백 연결:
    #   _goal_response_callback: Nav2가 목표 수락/거부했을 때 호출
    #   _result_callback: 이동 완료/실패했을 때 호출
    # 인자:
    #   x, y — 맵 기준 목표 좌표 (미터)
    #   yaw  — 도착 후 바라볼 방향 (라디안)
    # 반환값: (성공여부 bool, 결과 메시지 str)
    def navigate_to_goal(self, x, y, yaw):
        nav_log.info(f"Nav2 서버 연결 확인 중...")
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            nav_log.error("Nav2 서버 응답 없음")
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

        nav_log.info(f"목표 전송 → x={x:.2f}  y={y:.2f}  yaw={yaw:.2f}rad")
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_callback)
        return True, f"Nav2 이동 명령 전송 → x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}rad"

    # Nav2가 이동 목표를 수락하거나 거부했을 때 호출되는 콜백 함수.
    # 목표가 거부되는 주요 원인:
    #   - RViz에서 2D Pose Estimate로 초기 위치를 설정하지 않은 경우
    #   - 목표 좌표가 costmap 상의 장애물 안쪽인 경우
    #   - Nav2가 아직 완전히 초기화되지 않은 경우
    # 목표가 수락되면 _result_callback을 등록해서 완료/실패 여부를 추후 통보받는다.
    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            nav_log.warning("목표 거부됨 ✗  (초기 위치 설정 또는 맵 확인 필요)")
            return
        nav_log.info("목표 수락됨 ✓  이동 시작...")
        goal_handle.get_result_async().add_done_callback(self._result_callback)

    # Nav2 이동이 완료되거나 실패했을 때 호출되는 콜백 함수.
    # GoalStatus 값:
    #   STATUS_SUCCEEDED(4): 목표 지점에 정상 도달
    #   그 외: 경로를 못 찾거나, 장애물에 막히거나, 취소된 경우
    def _result_callback(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            nav_log.info("목표 도달 완료! ✓")
        else:
            nav_log.warning(f"이동 실패 ✗  (status={status})")

    # 로봇을 즉시 정지시키는 함수.
    # Nav2를 거치지 않고 /cmd_vel 토픽에 속도 0(Twist())을 직접 발행한다.
    # Twist()의 기본값이 linear.x=0, angular.z=0 이므로 로봇이 즉시 멈춘다.
    def stop(self):
        self.cmd_vel_pub.publish(Twist())
        nav_log.info("정지 명령 전송")


# RobotNode를 생성하고 백그라운드 스레드에서 ROS2 이벤트 루프를 시작하는 함수.
# app.py 시작 시 한 번 호출된다.
# rclpy.spin(node)를 별도 스레드에서 실행하는 이유:
#   spin()은 ROS2 토픽/액션 이벤트를 계속 감시하는 블로킹 루프다.
#   메인 스레드에서 실행하면 FastAPI 웹서버가 시작되지 못한다.
#   daemon=True로 설정하면 메인 프로세스(app.py) 종료 시 이 스레드도 자동 종료된다.
# 반환값: 초기화된 RobotNode 인스턴스
def create_node() -> RobotNode:
    rclpy.init()
    node = RobotNode()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
    return node
