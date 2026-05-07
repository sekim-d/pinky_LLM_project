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


@app.post("/command")
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


@app.get("/detected_markers")
async def detected_markers():
    markers = ros_node.detected_markers
    web_log.info(f"감지된 마커 조회 → {markers}")
    return {"markers": markers}


@app.post("/auto_navigate")
async def set_auto_navigate(req: AutoNavReq):
    ros_node.auto_navigate = req.enabled
    state = "ON" if req.enabled else "OFF"
    web_log.info(f"자동 이동 모드 {state}")
    return {"auto_navigate": ros_node.auto_navigate}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    with open(html_path, encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    web_log.info("Pinky 웹서버 시작 — http://0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
