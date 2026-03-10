import socket
import struct
import cv2
import numpy as np
import threading

frames = {
    "Cam1": None,
    "Cam2": None
}

def receive_camera(port, name):

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", port))
    server.listen(1)

    print("Waiting for camera on port", port)

    conn, addr = server.accept()
    print("Connected:", addr)

    data = b""

    while True:

        while len(data) < 4:
            packet = conn.recv(4096)
            if not packet:
                return
            data += packet

        packed_size = data[:4]
        data = data[4:]
        frame_size = struct.unpack("I", packed_size)[0]

        while len(data) < frame_size:
            data += conn.recv(4096)

        frame_data = data[:frame_size]
        data = data[frame_size:]

        frame = cv2.imdecode(
            np.frombuffer(frame_data, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is not None:
            frames[name] = frame


threading.Thread(target=receive_camera, args=(4000,"Cam1"), daemon=True).start()
threading.Thread(target=receive_camera, args=(4001,"Cam2"), daemon=True).start()


while True:

    if frames["Cam1"] is not None:
        cv2.imshow("Cam1", frames["Cam1"])

    if frames["Cam2"] is not None:
        cv2.imshow("Cam2", frames["Cam2"])

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()