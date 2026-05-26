import socket
import struct
import numpy as np
import cv2

SERVER_IP   = "192.168.1.65"
SERVER_PORT = 9000

# 패킷 헤더: frame_id(4B) + chunk_idx(2B) + total_chunks(2B) = 8B
HEADER_FMT  = ">IHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5.0)

    # 서버에 등록 신호 전송
    sock.sendto(b'\x00', (SERVER_IP, SERVER_PORT))
    print(f"[연결] {SERVER_IP}:{SERVER_PORT}")

    # frame_id → {chunk_idx: bytes}
    buf = {}

    try:
        while True:
            try:
                packet, _ = sock.recvfrom(65536)
            except socket.timeout:
                print("[경고] 수신 타임아웃 — 재연결 시도")
                sock.sendto(b'\x00', (SERVER_IP, SERVER_PORT))
                continue

            if len(packet) < HEADER_SIZE:
                continue

            frame_id, chunk_idx, total = struct.unpack(HEADER_FMT, packet[:HEADER_SIZE])
            chunk_data = packet[HEADER_SIZE:]

            if frame_id not in buf:
                buf[frame_id] = {}
            buf[frame_id][chunk_idx] = chunk_data

            # 청크가 전부 모이면 프레임 조립 후 표시
            if len(buf[frame_id]) == total:
                data  = b"".join(buf[frame_id][i] for i in range(total))
                frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    cv2.imshow("Pinky Camera", frame)
                del buf[frame_id]

            # 10프레임 이상 지난 미완성 프레임 정리
            for fid in [f for f in buf if f < frame_id - 10]:
                del buf[fid]

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
