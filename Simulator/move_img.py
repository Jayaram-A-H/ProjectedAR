import socket
import time

s = socket.socket()
s.connect(("127.0.0.1", 5005))

print("Connected to Unity!")

while True:
    x=input()
    x=x.split(',')
    i=0
    for xx in x:
        x[i]=float(xx)
        i+=1
    msg = f"move {float(x[0])} {float(x[1])} {float(x[2])} {float(x[3])} {float(x[4])} {float(x[5])}\n"
    s.sendall(msg.encode()) # move right
    time.sleep(0.1)