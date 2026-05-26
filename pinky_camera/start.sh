#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEDIAMTX_BIN="$SCRIPT_DIR/mediamtx"
MEDIAMTX_VER="v1.9.1"
MEDIAMTX_URL="https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VER}/mediamtx_${MEDIAMTX_VER}_linux_amd64.tar.gz"

if ! command -v ffmpeg &>/dev/null; then
  echo "[오류] FFmpeg 없음 → sudo apt install -y ffmpeg"
  exit 1
fi

if [ ! -f "$MEDIAMTX_BIN" ]; then
  echo "[MediaMTX] 다운로드 중... ($MEDIAMTX_VER)"
  TMP=$(mktemp -d)
  if command -v curl &>/dev/null; then
    curl -fsSL "$MEDIAMTX_URL" -o "$TMP/mediamtx.tar.gz"
  else
    wget -q "$MEDIAMTX_URL" -O "$TMP/mediamtx.tar.gz"
  fi
  tar -xzf "$TMP/mediamtx.tar.gz" -C "$TMP" mediamtx
  cp "$TMP/mediamtx" "$MEDIAMTX_BIN"
  chmod +x "$MEDIAMTX_BIN"
  rm -rf "$TMP"
fi

cleanup() {
  kill "$MEDIAMTX_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$MEDIAMTX_BIN" "$SCRIPT_DIR/mediamtx.yml" &
MEDIAMTX_PID=$!

HOST_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "스트림 주소: http://${HOST_IP}:8888/cam1/index.m3u8"
echo "종료: Ctrl+C"
echo ""

wait $MEDIAMTX_PID
