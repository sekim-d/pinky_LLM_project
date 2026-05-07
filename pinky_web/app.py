from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from llm_client import ask_llm, execute_cmd, parse_llm_response
from logger import web_log
from ros_node import create_node

ros_node = create_node()
app = FastAPI()


class CommandReq(BaseModel):
    text: str


class AutoNavReq(BaseModel):
    enabled: bool


# 브라우저에서 자연어 명령을 받아 처리하는 엔드포인트. POST /command
# 처리 순서:
#   1. ask_llm()으로 Ollama에 명령 전송 → JSON 문자열 응답 수신
#   2. parse_llm_response()로 JSON 문자열 → 딕셔너리 파싱
#   3. execute_cmd()로 딕셔너리 해석 → RobotNode 동작 실행
#   4. 결과를 브라우저에 반환
# 요청: {"text": "마커A로 가"}
# 응답: {"ok": true, "llm_json": {...}, "result": "Nav2 이동 명령 전송 ..."}
async def command(req: CommandReq):
    web_log.info(f"명령 수신 → \"{req.text}\"")
    try:
        raw    = ask_llm(req.text)
        cmd    = parse_llm_response(raw)
        result = execute_cmd(cmd, ros_node)
        web_log.info(f"처리 완료 → {result}")
        return {"ok": True, "llm_json": cmd, "result": result}
    except Exception as e:
        web_log.error(f"처리 실패 → {e}")
        return JSONResponse(status_code=500,
                            content={"ok": False, "error": str(e)})


# 카메라로 감지된 ArUco 마커 목록과 맵 좌표를 반환하는 엔드포인트. GET /detected_markers
# ros_node.detected_markers 딕셔너리를 그대로 반환한다.
# 이 딕셔너리는 _image_cb()에서 마커가 감지될 때마다 실시간으로 업데이트된다.
# 응답 예시: {"markers": {0: {"x": 0.501, "y": 1.298}, 1: {"x": -1.55, "y": -0.73}}}
@app.get("/detected_markers")
async def detected_markers():
    markers = ros_node.detected_markers
    web_log.info(f"감지된 마커 조회 → {markers}")
    return {"markers": markers}


# ArUco 마커 감지 시 자동 이동 모드를 켜고 끄는 엔드포인트. POST /auto_navigate
# True로 설정하면: 카메라가 마커를 감지하는 순간 자동으로 Nav2 이동 명령 전송
# False로 설정하면: 마커 감지는 계속하지만 이동 명령은 보내지 않음
# 요청: {"enabled": true}
# 응답: {"auto_navigate": true}
@app.post("/auto_navigate")
async def set_auto_navigate(req: AutoNavReq):
    ros_node.auto_navigate = req.enabled
    state = "ON" if req.enabled else "OFF"
    web_log.info(f"자동 이동 모드 {state}")
    return {"auto_navigate": ros_node.auto_navigate}


# 브라우저로 접속했을 때 웹 UI(index.html)를 반환하는 엔드포인트. GET /
# Path(__file__)로 현재 파일(app.py) 위치를 기준으로 index.html 경로를 구한다.
# 절대 경로를 하드코딩하지 않아서 어느 환경에서도 동작한다.
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    with open(html_path, encoding="utf-8") as f:
        return f.read()


app.post("/command")(command)

if __name__ == "__main__":
    web_log.info("Pinky 웹서버 시작 — http://0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
