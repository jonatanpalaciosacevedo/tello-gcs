from pyimagesearch.blur_detection.blur_detector import *
from imutils.video import VideoStream
import imutils
import time
import cv2
import pytesseract
from djitellopy import tello

drone = tello.Tello()
drone.connect()
print(drone.get_battery())
drone.streamon()  # turn on camera
time.sleep(2)

letter = "A"

# path of pytesseract execution folder (Change as needed)
# For windows, the path is usually in either of these two:
# r"C:\Users\YOUR_USER\AppData\Local\Tesseract-OCR\tesseract.exe"
# r"C:\Program Files\Tesseract-OCR\tessdata"
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

# initialize our output video writer along with the dimensions of the
# output frame
writer = None
outputW = None
outputH = None

# create a named window for our output OCR visualization (a named
# window is required here so that we can automatically position it
# on our screen)
cv2.namedWindow("Output")

# Get video from webcam / drone / phone
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

cntr = 0

# loop over frames from the video stream
while True:
    # Drone camera
    orig = drone.get_frame_read().frame

    # grab the next frame and handle if we are reading from either
    # a webcam or a video file
    cntr = cntr + 1

    # resize the frame and compute the ratio of the *new* width to
    # the *old* width
    frame = imutils.resize(orig, height=480, width=640)

    # convert the frame to grayscale and detect if the frame is
    # considered blurry or not
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (mean, blurry) = detect_blur_fft(gray, thresh=11)

    # Testing
    img_h, img_w, _ = frame.shape
    frame = cv2.circle(frame, (320, 180), radius=3, color=(0, 0, 255), thickness=-1)

    # process data if not blurry
    if not blurry:
        # Test detection threshold
        # color = (0, 255, 0)
        text = "Detecting ({:.4f})"
        text = text.format(mean)
        print(text)
        # cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.7, color, 2)

        # Process letters here
        if (cntr % 20) == 0:
            img_h, img_w, _ = frame.shape
            x1, y1, w1, h1 = 0, 0, img_h, img_w
            imgchar = pytesseract.image_to_string(frame)
            imgboxes = pytesseract.image_to_boxes(frame)

            if letter in imgchar:
                print(f"found the letter {letter}")

            print(imgchar)

            for boxes in imgboxes.splitlines():
                boxes = boxes.split(" ")
                x, y, w, h = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
                cv2.rectangle(frame, (x, img_h - y), (w, img_h - h), (0, 0, 255), 1)

    # show the output video OCR visualization
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

vs.stop()

# close any open windows
cv2.destroyAllWindows()
