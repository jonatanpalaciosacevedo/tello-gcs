from djitellopy import tello

drone = tello.Tello()
drone.connect()
print(f"Battery: {drone.get_battery()}%")


"""
import tellopy
import av
import cv2
import time
import numpy as np

drone = tellopy.Tello()
drone.connect()
drone.wait_for_connection(10)
drone.start_video()

container = av.open(drone.get_video_stream())
frame_skip = 300

while True:
    for frame in container.decode(video=0):
        if 0 < frame_skip:
            frame_skip = frame_skip - 1
            continue
        start_time = time.time()
        image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
        cv2.imshow('Original', image)

        cv2.waitKey(1)
        if frame.time_base < 1.0/60:
            time_base = 1.0/60
        else:
            time_base = frame.time_base
        frame_skip = int((time.time() - start_time)/time_base)
"""