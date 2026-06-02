"""
Task Manager - ROS2 Action 기반 물류창고 로봇 오케스트레이터
포트: 8090 (태스크 생성 HTTP — LLM 웹서버용)

send_goal_async()는 반드시 executor 스레드 안에서 호출해야 응답 콜백이 동작함.
타이머 콜백(50ms) 안에서 큐를 꺼내 goal을 전송하는 방식으로 이를 보장함.
"""

import json
import queue
import threading
import uuid
from datetime import datetime
from enum import Enum

import rclpy
import uvicorn
from fastapi import FastAPI, HTTPException
from msgs.action import RobotCommand
from pydantic import BaseModel
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

# ── 로봇 설정 ──────────────────────────────────────────────────
ROBOTS: dict[str, dict] = {
    "jetcobot1": {"type": "jetcobot", "status": "idle", "task_id": None},
    "jetcobot2": {"type": "jetcobot", "status": "idle", "task_id": None},
    "pinky1":    {"type": "pinky",    "status": "idle", "task_id": None},
}

ZONES: dict[str, str] = {f"zone_{i}": "free" for i in range(1, 17)}

# ── 태스크 상태 머신 ───────────────────────────────────────────
class TaskState(str, Enum):
    PENDING             = "pending"
    JETCOBOT1_PICKUP    = "jetcobot1_pickup"
    PINKY_TO_LOADING    = "pinky_to_loading"
    JETCOBOT1_LOADING   = "jetcobot1_loading"
    PINKY_TO_STORAGE    = "pinky_to_storage"
    JETCOBOT2_UNLOADING = "jetcobot2_unloading"
    PINKY_TO_HOME       = "pinky_to_home"
    DONE                = "done"
    FAILED              = "failed"

class Task:
    def __init__(self, task_id: str, storage_zone: str):
        self.task_id      = task_id
        self.storage_zone = storage_zone
        self.state        = TaskState.PENDING
        self.jetcobot1_id = "jetcobot1"
        self.jetcobot2_id = "jetcobot2"
        self.pinky_id: str | None = None
        self.created_at   = datetime.now().isoformat()
        self.updated_at   = datetime.now().isoformat()
        self.log: list[str] = []

    def record(self, msg: str):
        self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        self.updated_at = datetime.now().isoformat()
        print(f"[Task {self.task_id[:8]}] {msg}")

    def to_dict(self) -> dict:
        return {
            "task_id":      self.task_id,
            "state":        self.state,
            "storage_zone": self.storage_zone,
            "pinky":        self.pinky_id,
            "created_at":   self.created_at,
            "updated_at":   self.updated_at,
            "log":          self.log,
        }

# ── 전역 저장소 ────────────────────────────────────────────────
tasks: dict[str, Task] = {}
_lock = threading.Lock()
ros_node: "TaskManagerNode" = None
_cmd_queue: queue.Queue = queue.Queue()

# ── ROS2 액션 클라이언트 노드 ──────────────────────────────────
class TaskManagerNode(Node):
    def __init__(self):
        super().__init__("task_manager")

        self._cb_group = ReentrantCallbackGroup()
        self._action_clients = {
            robot_id: ActionClient(self, RobotCommand, f"/{robot_id}/command",
                                   callback_group=self._cb_group)
            for robot_id in ROBOTS
        }
        # executor 스레드 안에서 50ms마다 큐 확인 후 goal 전송
        self.create_timer(0.05, self._flush_cmd_queue, callback_group=self._cb_group)
        self.get_logger().info("Task Manager Action Node 시작")

    def _flush_cmd_queue(self):
        try:
            cmd = _cmd_queue.get_nowait()
        except queue.Empty:
            return

        robot_id   = cmd["robot_id"]
        action     = cmd["action"]
        parameters = cmd["parameters"]
        task_id    = cmd["task_id"]
        client     = self._action_clients[robot_id]

        if not client.server_is_ready():
            self._retry = getattr(self, "_retry", {})
            self._retry[robot_id] = self._retry.get(robot_id, 0) + 1
            if self._retry[robot_id] % 40 == 1:  # 50ms × 40 = 2초마다
                print(f"[WARN] {robot_id} 서버 미준비 — 대기 중")
            _cmd_queue.put(cmd)
            return
        self._retry = getattr(self, "_retry", {})
        self._retry[robot_id] = 0

        goal = RobotCommand.Goal()
        goal.action_type     = action
        goal.parameters_json = json.dumps(parameters)
        goal.task_id         = task_id

        future = client.send_goal_async(goal)
        future.add_done_callback(
            lambda f: self._on_goal_response(f, robot_id, task_id)
        )
        print(f"[ROS2 Action] → /{robot_id}/command  action={action}  task={task_id[:8]}")

    def _on_goal_response(self, future, robot_id: str, task_id: str):
        print(f"[GOAL_RESPONSE] {robot_id} 응답 수신 (task={task_id[:8]})")
        goal_handle = future.result()
        if not goal_handle.accepted:
            print(f"[WARN] {robot_id} 목표 거부됨")
            task = tasks.get(task_id)
            if task:
                _fail_task(task, f"{robot_id} 목표 거부됨")
            return

        print(f"[GOAL_RESPONSE] {robot_id} 목표 수락됨 — result 대기 중")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._on_result(f, robot_id, task_id)
        )

    def _on_result(self, future, robot_id: str, task_id: str):
        result  = future.result().result
        event   = result.event
        message = result.message
        print(f"[RESULT] robot={robot_id}  event={event!r}  message={message!r}  task={task_id[:8]}")
        with _lock:
            task = tasks.get(task_id)
            if task is None:
                return
            task.record(f"{robot_id} → {event}")
            _advance(task, robot_id, event)

# ── 헬퍼 ──────────────────────────────────────────────────────
def _find_idle_pinky() -> str | None:
    for rid, r in ROBOTS.items():
        if r["type"] == "pinky" and r["status"] == "idle":
            return rid
    return None

def _set_robot_busy(robot_id: str, task_id: str):
    ROBOTS[robot_id]["status"]  = "busy"
    ROBOTS[robot_id]["task_id"] = task_id

def _set_robot_idle(robot_id: str):
    ROBOTS[robot_id]["status"]  = "idle"
    ROBOTS[robot_id]["task_id"] = None

def _fail_task(task: Task, reason: str):
    task.state = TaskState.FAILED
    task.record(f"태스크 실패 — {reason}")
    for robot_id in [task.jetcobot1_id, task.jetcobot2_id]:
        if ROBOTS[robot_id]["task_id"] == task.task_id:
            _set_robot_idle(robot_id)
    if task.pinky_id and ROBOTS[task.pinky_id]["task_id"] == task.task_id:
        _set_robot_idle(task.pinky_id)

def _send_command(robot_id: str, action: str, parameters: dict, task_id: str):
    _cmd_queue.put({"robot_id": robot_id, "action": action,
                    "parameters": parameters, "task_id": task_id})

# ── 상태 전이 엔진 ────────────────────────────────────────────
def _advance(task: Task, robot_id: str, event: str):
    s = task.state

    if s == TaskState.JETCOBOT1_PICKUP and robot_id == task.jetcobot1_id and event == "done":
        _set_robot_idle(task.jetcobot1_id)
        pinky_id = _find_idle_pinky()
        if pinky_id is None:
            task.record("유휴 pinky 없음 — 대기 중")
            return
        task.pinky_id = pinky_id
        task.state    = TaskState.PINKY_TO_LOADING
        _set_robot_busy(pinky_id, task.task_id)
        task.record(f"{pinky_id} → 입고구역 이동 명령")
        _send_command(pinky_id, "navigate", {"location": "loading_zone"}, task.task_id)

    elif s == TaskState.PINKY_TO_LOADING and robot_id == task.pinky_id and event == "arrived":
        task.state = TaskState.JETCOBOT1_LOADING
        _set_robot_busy(task.jetcobot1_id, task.task_id)
        task.record(f"jetcobot1 → {task.pinky_id}에 물건 적재 명령")
        _send_command(task.jetcobot1_id, "load", {"target_pinky": task.pinky_id}, task.task_id)

    elif s == TaskState.JETCOBOT1_LOADING and robot_id == task.jetcobot1_id and event == "done":
        _set_robot_idle(task.jetcobot1_id)
        task.state = TaskState.PINKY_TO_STORAGE
        task.record(f"{task.pinky_id} → {task.storage_zone} 이동 명령")
        _send_command(task.pinky_id, "navigate", {"location": task.storage_zone}, task.task_id)

    elif s == TaskState.PINKY_TO_STORAGE and robot_id == task.pinky_id and event == "arrived":
        task.state = TaskState.JETCOBOT2_UNLOADING
        _set_robot_busy(task.jetcobot2_id, task.task_id)
        task.record(f"jetcobot2 → {task.storage_zone} 하역 명령")
        _send_command(task.jetcobot2_id, "unload",
                      {"zone": task.storage_zone, "source_pinky": task.pinky_id}, task.task_id)

    elif s == TaskState.JETCOBOT2_UNLOADING and robot_id == task.jetcobot2_id and event == "done":
        _set_robot_idle(task.jetcobot2_id)
        ZONES[task.storage_zone] = "occupied"
        task.state = TaskState.PINKY_TO_HOME
        task.record(f"{task.pinky_id} → 홈으로 복귀 명령")
        _send_command(task.pinky_id, "navigate", {"location": "home"}, task.task_id)

    elif s == TaskState.PINKY_TO_HOME and robot_id == task.pinky_id and event == "arrived":
        _set_robot_idle(task.pinky_id)
        task.state = TaskState.DONE
        task.record(f"태스크 완료 — {task.storage_zone} 점유, {task.pinky_id} 홈 복귀")

    elif event == "error":
        _fail_task(task, f"{robot_id} 오류 발생")

    else:
        task.record(f"[무시] {robot_id} / {event} (현재 상태: {s})")

# ── FastAPI ────────────────────────────────────────────────────
app = FastAPI(title="Task Manager")

class TaskRequest(BaseModel):
    storage_zone: str

@app.post("/task")
def create_task(req: TaskRequest):
    if req.storage_zone not in ZONES:
        raise HTTPException(400, f"유효하지 않은 구역: {req.storage_zone}")
    if ZONES[req.storage_zone] == "occupied":
        raise HTTPException(409, f"{req.storage_zone} 이미 점유 중")
    if ROBOTS["jetcobot1"]["status"] != "idle":
        raise HTTPException(503, "jetcobot1 사용 중")

    task_id = str(uuid.uuid4())
    task = Task(task_id, req.storage_zone)
    with _lock:
        tasks[task_id] = task

    task.state = TaskState.JETCOBOT1_PICKUP
    _set_robot_busy("jetcobot1", task_id)
    task.record(f"태스크 시작 → jetcobot1 물건 집기 (목표: {req.storage_zone})")
    _send_command("jetcobot1", "pickup", {}, task_id)

    return {"task_id": task_id, "state": task.state}

@app.get("/task/{task_id}")
def get_task(task_id: str):
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(404)
    return task.to_dict()

@app.get("/tasks")
def list_tasks():
    return [t.to_dict() for t in tasks.values()]

@app.get("/robots")
def get_robots():
    return ROBOTS

@app.get("/zones")
def get_zones():
    return ZONES

@app.get("/debug")
def debug():
    return {
        "robots": ROBOTS,
        "zones":  {k: v for k, v in ZONES.items() if v != "free"},
        "tasks":  [
            {"task_id": t.task_id[:8], "state": t.state, "pinky": t.pinky_id, "log": t.log[-5:]}
            for t in tasks.values()
        ],
    }

# ── 진입점 ────────────────────────────────────────────────────
def ros_spin():
    try:
        executor = MultiThreadedExecutor()
        executor.add_node(ros_node)
        print("[ROS2] executor 시작")
        executor.spin()
        print("[ROS2] executor 종료됨")
    except Exception as e:
        import traceback
        print(f"[ERROR] ros_spin 크래시: {e}")
        traceback.print_exc()

def main():
    global ros_node
    rclpy.init()
    ros_node = TaskManagerNode()

    threading.Thread(target=ros_spin, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8090)
