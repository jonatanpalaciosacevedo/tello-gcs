from pyimagesearch.blur_detection.blur_detector import *
import imutils
import time
import cv2
import pytesseract
from djitellopy import tello
import os


# ---------------
# In case of self installation of tesseract:
# Path of pytesseract execution folder (Change as needed)

# For windows, the path is either here:
# your_username = os.getlogin()
# r"C:\Users\your_username\AppData\Local\Tesseract-OCR\tesseract.exe"

# Or here:
# r"C:\Program Files\Tesseract-OCR\tessdata"
# ---------------

# For Windows, the folder is already installed in this project (easier):
# pytesseract.pytesseract.tesseract_cmd = r'../modules/tesseractwin/tesseract.exe'

# For mac users you have to install tesseract using Homebrew and the path is usually here:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

# Or here:
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/5.2.0/bin/tesseract'  # Depending on the version of tesseract


# Connect via AP Mode
drone = tello.Tello()

drone.connect()
print(f"Battery: {drone.get_battery()}%")
drone.streamon()  # turn on camera
frame_read = drone.get_frame_read()

# Best resolution
# drone.set_video_resolution(drone.RESOLUTION_720P)

# Most FPS
# drone.set_video_fps(drone.FPS_30)

# create a named window for our output OCR visualization (a named
# window is required here so that we can automatically position it
# on our screen)
cv2.namedWindow("Output")

# Get video from webcam / drone / phone
print("[INFO] starting video stream...")
time.sleep(2.0)

cntr = 0
detecting = False

drone.takeoff()

# loop over frames from the video stream
while True:
    # Drone camera
    orig = frame_read.frame

    # grab the next frame and handle if we are reading from either
    # a webcam or a video file
    cntr = cntr + 1

    key = cv2.waitKey(1) & 0xFF

    # resize the frame and compute the ratio of the *new* width to
    # the *old* width
    frame = imutils.resize(orig, height=480, width=640)

    # convert the frame to grayscale and detect if the frame is
    # considered blurry or not
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (mean, blurry) = detect_blur_fft(gray, thresh=7)

    # Show stats
    color = (0, 255, 0)
    thr = "Thr ({:.4f})"
    thr = thr.format(mean)
    temp = f"Temperature: {drone.get_temperature()}C"
    bat = f"Battery: {drone.get_battery()}%"

    """cv2.putText(frame, thr, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, temp, (10, 25 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, bat, (10, 25 + 40 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)"""

    # if "c" is press, start detecting
    if key == ord("c"):
        detecting = not detecting

    if key == ord("w"):
        drone.move_up(20)

    if key == ord("s"):
        drone.move_down(20)

    if key == ord("d"):
        drone.move_right(20)

    if key == ord("a"):
        drone.move_left(20)

    # process data if not blurry
    if not blurry and detecting:
        # Test detection threshold
        text = "Detecting..."

        """cv2.putText(frame, text, (10, 25 + 40 + 40 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)"""

        # Process letters here
        if (cntr % 20) == 0:
            img_h, img_w, _ = frame.shape
            x1, y1, w1, h1 = 0, 0, img_h, img_w
            imgchar = pytesseract.image_to_string(frame, lang="eng+equ", config="--psm 10")
            imgboxes = pytesseract.image_to_boxes(frame)

            for boxes in imgboxes.splitlines():
                boxes = boxes.split(" ")
                x, y, w, h = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
                cv2.rectangle(frame, (x, img_h - y), (w, img_h - h), (0, 0, 255), 1)

            print(imgchar)

    # show the output video OCR visualization
    cv2.imshow("Output", frame)

    # if the 'q' key was pressed, break from the loop
    if key == ord("q"):
        drone.land()
        drone.streamoff()
        break


# close any open windows
cv2.destroyAllWindows()
