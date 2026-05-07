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
from sensor_msgs.msg import Image
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from config import CAMERA_MATRIX, DIST_COEFFS, MARKER_OBJ_PTS
from logger import nav_log, aruco_log

_aruco_dict    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_aruco_params  = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)


class RobotNode(Node):
    def __init__(self):
        super().__init__('web_controller')

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

    # ── ArUco ─────────────────────────────────────────
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
            _, rvec, tvec = cv2.solvePnP(
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

    # ── 네비게이션 ─────────────────────────────────────
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

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            nav_log.warning("목표 거부됨 ✗  (초기 위치 설정 또는 맵 확인 필요)")
            return
        nav_log.info("목표 수락됨 ✓  이동 시작...")
        goal_handle.get_result_async().add_done_callback(self._result_callback)

    def _result_callback(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            nav_log.info("목표 도달 완료! ✓")
        else:
            nav_log.warning(f"이동 실패 ✗  (status={status})")

    def stop(self):
        self.cmd_vel_pub.publish(Twist())
        nav_log.info("정지 명령 전송")


def create_node() -> RobotNode:
    rclpy.init()
    node = RobotNode()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
    return node
