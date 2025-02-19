import cv2
import time
from djitellopy import tello
import imutils

# Connect to drone via AP (Specify your drone IP)
tello_ip = "192.168.0.31"  # Tello IP address

# Or connect via AP Mode
# drone = tello.Tello()

drone = tello.Tello(host=tello_ip)
drone.connect()
print(f"Battery: {drone.get_battery()}%")
drone.streamon()  # turn on camera

# Get bottom camera
drone.set_video_direction(drone.CAMERA_DOWNWARD)
time.sleep(1)

# Resolution
# drone.set_video_resolution(drone.RESOLUTION_720P)

# FPS
# drone.set_video_fps(drone.FPS_30)

# Height
# print(drone.get_height())
# Gives the height relative to previous position, if not moving its 0

while True:
    img = drone.get_frame_read().frame
    img = imutils.resize(img, width=640, height=480)

    cv2.imshow("Image", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


