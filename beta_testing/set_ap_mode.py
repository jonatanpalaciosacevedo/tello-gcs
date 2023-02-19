import socket
import argparse
import sys


def get_socket():
    """
    Gets a socket.
    :return: Socket.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('', 8889))

    return s


def set_ap(ssid, password, address):
    """
    A Function to set tello in Access Point (AP) mode.

    :param ssid: The SSID of the network (e.g. name of the Wi-Fi).
    :param password: The password of the network.
    :param address: Tello IP.
    :return: None.
    """
    s = get_socket()

    cmd = 'command'
    print(f'sending cmd {cmd}')
    s.sendto(cmd.encode('utf-8'), address)

    response, ip = s.recvfrom(100)
    print(f'from {ip}: {response}')

    cmd = f'ap {ssid} {password}'
    print(f'sending cmd {cmd}')
    s.sendto(cmd.encode('utf-8'), address)

    response, ip = s.recvfrom(100)
    print(f'from {ip}: {response}')


def parse_args(args):
    """
    Parses arguments.
    :param args: Arguments.
    :return: Arguments.
    """
    parser = argparse.ArgumentParser(description="Set Tello EDU Drone to AP Mode")

    parser.add_argument('-s', '--ssid', help='SSID', required=True)
    parser.add_argument('-p', '--pwd', help='password', required=True)
    parser.add_argument('--ip', help='Tello IP', default='192.168.10.1', required=False)
    parser.add_argument('--port', help='Tello port', default=8889, type=int, required=False)

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    ssid = args.ssid
    pwd = args.pwd
    tello_address = (args.ip, args.port)

    set_ap(ssid, pwd, tello_address)
