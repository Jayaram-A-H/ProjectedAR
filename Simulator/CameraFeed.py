import socket
import struct
import cv2
import numpy as np

HOST = "0.0.0.0"
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print("Waiting for Unity connection...")

conn, addr = server.accept()
print("Connected:", addr)

data = b""

while True:

    # receive frame size
    while len(data) < 4:
        packet = conn.recv(4096)
        if not packet:
            break
        data += packet

    packed_size = data[:4]
    data = data[4:]
    frame_size = struct.unpack("I", packed_size)[0]

    # receive frame data
    while len(data) < frame_size:
        data += conn.recv(4096)

    frame_data = data[:frame_size]
    data = data[frame_size:]

    # decode image
    frame = cv2.imdecode(
        np.frombuffer(frame_data, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        continue

    cv2.imshow("Unity Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

conn.close()
cv2.destroyAllWindows()