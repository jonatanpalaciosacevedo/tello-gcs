import socket
import ipaddress
from subprocess import Popen, PIPE
from djitellopy import tello
import numpy as np


class DroneDB:
    def __init__(self, serial_number):

        ids = {"unknown": 1,
               "0TQDG1GEDB258Z": 2,
               "0TQZH4EED00478": 4}

        self.serial_number = serial_number
        self.db = "drones.dat"
        self.id = ids[self.serial_number]

    def add_drone(self):
        with open(self.db, "a") as f:
            f.write(str(self.serial_number) + "\n")

    def in_db(self):
        with open(self.db, "r") as f:
            found = False
            for line in f.readlines():
                if self.serial_number == line.strip():
                    found = True
                    print("Drone in Data Base!")

            if not found:
                print("Drone not in Data Base!")

        return found


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


def detect_drones_with_ip(list_of_ips):
    # Connect to drone via AP (Specify your drone IP)
    drones = []

    for ip in list_of_ips:
        drone = tello.Tello(host=ip)

        # Connect with UDP commands
        if "Aborting command" not in drone.send_command_with_return("command"):
            # print("RESPONSE: " + drone.send_command_with_return("command"))
            # print("Drones detected: " + str(i))
            drones.append(drone)

    # Found valid drones within the ip list
    if drones:
        print(f"Found {len(drones)} drones.")

        for drone in drones:
            drone_i = DroneDB(drone.send_command_with_return("sn?"))
            if not drone_i.in_db():
                print("Drone not in DB, adding it now")
                drone_i.add_drone()
            else:
                print(f"Hello {drone_i.id}!")

    else:
        print("No drones found")
        return None

    return drones


ip_s = ["192.168.0.30", "192.168.0.31"]
droness = detect_drones_with_ip(ip_s)

# print(lis)
# print(lis[2].get_battery())  # second found drone




# Get list with all ip addresses that are connected to network (excluding the ip address of current device)
# ip_list = scan_ips()

# print(ip_list)

# obj = DroneDB(10)
# save_object(obj)


"""

    - Open drone DB
    - Scan ips connected to network
    - If a drone is in DB
        - Send UDP command sn? 
        - If serial np is in DB, say hello 
    - If it finds an ip not in DB, ask to register it?
    - Register a drone if you want
    - Save drone to DB


"""