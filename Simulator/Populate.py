import socket
import json

coords = [
    {"x": 0, "y": 0, "z": 0},
    {"x": 2, "y": 0, "z": 1},
    {"x": -3, "y": 0, "z": 5}
]
s_obj = socket.socket() 
s_obj.connect(("127.0.0.1", 5009))
data = json.dumps({"points": coords}) + "\n"  # newline = message end
print(data)
s_obj.sendall(data.encode())