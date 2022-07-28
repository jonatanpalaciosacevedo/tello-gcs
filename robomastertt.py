import socket
import sys

host = "192.168.10.1"
port = 8889

address = (host, int(port))

# Establish a UDP connection with the control command port of the robot.
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print("Connecting...")

s.connect(address)
print("Connected!")

num_msg = 0

# Send messages
while True:

    # Commands
    if num_msg == 0:
        msg = "command"
    elif num_msg == 1:
        msg = "mon"
    elif num_msg == 2:
        msg = "mdirection 0"
    elif num_msg == 3:
        msg = "logon"
    elif num_msg == 4:
        # Heart 1
        msg = "EXT mled g 000000000rr0rr00rrrrrrr0rrrrrrr0rrrrrrr00rrrrr0000rrr000000r0000"
    elif num_msg == 5:
        # Heart 2
        msg = "EXT mled g 000000000rr00rr0rrrrrrrrrrrrrrrr0rrrrrr000rrrr00000rr00000000000"


    # Send control commands to the robot.
    s.send(msg.encode('utf-8'))

    try:
        # Wait for the robot to return the execution result.
        buf = s.recv(1024)

        print(buf.decode('utf-8'))
    except socket.error as e:
        print("Error receiving :", e)
        sys.exit(1)
    if not len(buf):
        break

    num_msg = num_msg + 1

# Disconnect the port connection.
s.shutdown(socket.SHUT_WR)
s.close()




