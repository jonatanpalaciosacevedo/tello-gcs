from pyimagesearch.blur_detection.blur_detector import *
import imutils
import time
import cv2
import pytesseract
import requests


# path of pytesseract execution folder (Change as needed)
# For windows, the path is usually in either of these two:
# r"C:\Users\YOUR_USER\AppData\Local\Tesseract-OCR\tesseract.exe"
# r"C:\Program Files\Tesseract-OCR\tessdata"

# For mac it can also be here:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/5.2.0/bin/tesseract'

# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
url = "http://192.168.0.22:8080/shot.jpg"

# create a named window for our output OCR visualization (a named
# window is required here so that we can automatically position it
# on our screen)
cv2.namedWindow("Output")

# Get video from webcam / drone / phone
print("[INFO] starting video stream...")
time.sleep(2.0)

cntr = 0
detecting = False

# loop over frames from the video stream
while True:
    # Phone camera with IP Camera
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)

    # grab the next frame and handle if we are reading from either
    # a webcam or a video file
    cntr = cntr + 1

    key = cv2.waitKey(1) & 0xFF

    # resize the frame and compute the ratio of the *new* width to
    # the *old* width
    frame = imutils.resize(img, height=480, width=640)

    # convert the frame to grayscale and detect if the frame is
    # considered blurry or not
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (mean, blurry) = detect_blur_fft(gray, thresh=7)

    # Press "c" to toggle detection
    # Aqui cambiar a que detecting sea cuando no se esta moviendo el dron en vez de la letra "c"
    if key == ord("c"):
        detecting = not detecting

    # process data if not blurry
    if not blurry and detecting:
        # Test detection threshold
        # color = (0, 255, 0)
        # text = "Detecting ({:.4f})"
        # text = text.format(mean)
        # print(text)

        # Show in window
        # cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.7, color, 2)

        # Process letters here
        if (cntr % 20) == 0:
            img_h, img_w, _ = frame.shape
            x1, y1, w1, h1 = 0, 0, img_h, img_w
            imgchar = pytesseract.image_to_string(frame, lang="eng+equ", config="--psm 10")
            # imgchar = pytesseract.image_to_string(frame, lang='equ', config='--psm 1 --oem 3')

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
        break


# close any open windows
cv2.destroyAllWindows()
