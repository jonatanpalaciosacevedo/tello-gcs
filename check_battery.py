from djitellopy import tello

drone = tello.Tello()
drone.connect()
print(f"Battery: {drone.get_battery()}%")
# drone.land()

