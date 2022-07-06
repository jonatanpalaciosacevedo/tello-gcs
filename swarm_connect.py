import socket
import ipaddress
from subprocess import Popen, PIPE
from djitellopy import tello
import time
import pickle


class DroneDB:
    def __init__(self, serial_number):
        self.serial_number = serial_number


def save_object(item):
    try:
        with open("data.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def extract_my_ip():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        st.connect(('10.255.255.255', 1))
        IP = st.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        st.close()
    return IP


def scan_ips():
    print("Scanning...")
    start_ip = extract_my_ip() + "/24"
    ip_net = ipaddress.ip_network(start_ip, strict=False)

    list_ips = []
    not_found = 0
    found = False

    # Loop through the connected hosts
    for ip in ip_net.hosts():
        # Convert the ip to a string so it can be used in the ping method
        ip = str(ip)

        if ip == extract_my_ip():
            found = True
            continue

        if found:
            # Let's ping the IP to see if it's online
            toping = Popen(['ping', '-c', '1', '-W', '50', ip], stdout=PIPE)
            output = toping.communicate()[0]
            hostalive = toping.returncode

            # Print whether or not device is online
            if hostalive == 0:
                list_ips.append(ip)
                not_found = 0
            else:
                not_found = not_found + 1

            if not_found > 30:
                return list_ips

    return list_ips


# Get list with all ip addresses that are connected to network (excluding the ip address of current device)
ip_list = scan_ips()

print(ip_list)

obj = DroneDB(10)
save_object(obj)
