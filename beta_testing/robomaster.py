import socket
import sys

# In direct connection mode, the default IP address of the robot is 192.168.2.1 and the control command port is port 40923.
host = "192.168.10.1"
port = 8889


def main():
    address = (host, int(port))

    # Establish a TCP connection with the control command port of the robot.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("Connecting...")

    s.connect(address)

    print("Connected!")

    while True:

        # Wait for the user to enter control commands.
        msg = input(">>> please input SDK cmd: ")

        # When the user enters Q or q, exit the current program.
        if msg.upper() == 'Q':
            break

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

    # Disconnect the port connection.
    s.shutdown(socket.SHUT_WR)
    s.close()


if __name__ == '__main__':
    main()



"""
command
mon
mdirection 0
logon
EXT led 255 255 255  # turn on top led
EXT mled l r 2.5 JON  # write a text
EXT mled g 000000000rr00rr0rrrrrrrrrrrrrrrr0rrrrrr000rrrr00000rr00000000000   # heartgrid
EXT mled g 00pppp000p0000p0p0p00p0pp000000pp0p00p0pp00pp00p0p0000p000pppp00   # smiley
EXT mled g 00pppp000p0000p0p0p00p0pp000000pp00pp00pp0p00p0p0p0000p000pppp00   # sad
EXT mled s p I   # write a letter
EXT mled sg     # make a starting pattern
EXT mled u g 2.5 0000b00bbb0b0b000b00b00000bb0000000b0000bbb00bbb000b0b0b0b00b0b0   # moving screen (up)

"""