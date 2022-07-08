import socket
import ipaddress
from subprocess import Popen, PIPE
from djitellopy import tello
import time


class DroneDB:
    def __init__(self, me, serial_number):

        ids = {"1111": 1,
               "0TQDG1GEDB258Z": 2,
               "3333": 3,
               "0TQZH4EED00478": 4,
               "5555": 5,
               "6666": 6,
               "7777": 7}

        self.me = me
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
                    print("Drone in Data Base.")

            if not found:
                print("Drone not in Data Base.")

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
    dict_drones = {}

    for ip in list_of_ips:
        drone = tello.Tello(host=ip)

        # Connect with UDP commands
        if "Aborting command" not in drone.send_command_with_return("command"):
            # print("RESPONSE: " + drone.send_command_with_return("command"))
            # print("Drones detected: " + str(i))
            drones.append(drone)
            time.sleep(1)

    # Found valid drones within the ip list
    if drones:
        print(f"Found {len(drones)} drones.")

        for drone in drones:
            drone_i = DroneDB(drone, drone.send_command_with_return("sn?"))
            time.sleep(1)
            if not drone_i.in_db():
                print("Drone not in DB, adding it now")
                drone_i.add_drone()
            else:
                print(f"Hello TELLO-{drone_i.id}!")

            dict_drones[drone_i.id] = drone_i
            print(f"TELLO-{drone_i.id} battery: {drone_i.me.get_battery()}%")
    else:
        print("No drones found")
        return None

    return dict_drones


def main():
    pass


if __name__ == "__main__":
    """
        1. Scan ips connected to network and get the list
        2. If a drone is in DB
            - If serial number is in DB, say hello
            - If not, register it
    """

    # Scan ips and get list
    # ips = scan_ips()

    # Or if you know the ips, type them directly on a list
    ips = ["192.168.0.30", "192.168.0.31"]

    # Check which of the ips is a drone and check db
    drones = detect_drones_with_ip(ips)  # Dictionary "drones" with drone number as key and DroneDB object as value

    for i in range(len(drones) - 1):
        print(f"Drone {drones[i].id} battery: {drones[i].get_battery()}")


