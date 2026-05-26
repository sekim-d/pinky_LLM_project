import socket
import struct
import threading
import time
import cv2

HOST         = "0.0.0.0"
PORT         = 9000
CAMERA_INDEX = 0
JPEG_QUALITY = 80
FPS_LIMIT    = 30
CHUNK_SIZE   = 60000  # UDP 패킷당 페이로드 최대 크기 (bytes)

# 패킷 헤더: frame_id(4B) + chunk_idx(2B) + total_chunks(2B) = 8B
HEADER_FMT  = ">IHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, PORT))
    print(f"[서버] UDP 대기 중 → {HOST}:{PORT}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[오류] 카메라 {CAMERA_INDEX} 열기 실패")
        return

    clients = set()
    lock = threading.Lock()

    def receiver():
        # 클라이언트가 패킷 보내면 목록에 등록
        while True:
            try:
                _, addr = sock.recvfrom(16)
                with lock:
                    if addr not in clients:
                        clients.add(addr)
                        print(f"[등록] {addr}")
            except Exception:
                pass

    threading.Thread(target=receiver, daemon=True).start()

    frame_id = 0
    interval = 1.0 / FPS_LIMIT

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            continue

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        data = jpeg.tobytes()

        # 큰 프레임은 청크로 분할
        chunks = [data[i:i + CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]
        total  = len(chunks)

        with lock:
            dead = set()
            for addr in clients:
                try:
                    for idx, chunk in enumerate(chunks):
                        header = struct.pack(HEADER_FMT, frame_id & 0xFFFFFFFF, idx, total)
                        sock.sendto(header + chunk, addr)
                except Exception:
                    dead.add(addr)
            clients -= dead
            for addr in dead:
                print(f"[제거] {addr} 연결 끊김")

        frame_id += 1
        sleep_t = interval - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)


if __name__ == "__main__":
    main()
