import time
import imutils
import cv2
from djitellopy import tello
from modules import pose_module as pm
from simple_pid import PID


def conversion(old_value):
    # OldRange = (OldMax - OldMin)
    # NewRange = (NewMax - NewMin)
    # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin

    old_max = 100
    old_min = -100
    new_max = 500
    new_min = 20
    old_range = old_max - old_min
    new_range = new_max - new_min

    new_value = ((old_value - old_min) * new_range / old_range) + new_min

    return new_value


class TelloController():
    def __init__(self):
        self.flying = False
        self.detector = pm.PoseDetector()
        self.detecting = False




# Ask if you want to take off the drone
flying = False
drone_camera = False

pid_roll = PID(0.25, 0, 0, setpoint=0, output_limits=(-100, 100))
pid_throttle = PID(0.4, 0, 0, setpoint=0, output_limits=(-100, 100))
pid_pitch = PID(0.4, 0, 0, setpoint=0, output_limits=(-100, 100))

# Init
detector = pm.PoseDetector()
detecting = False

# If drone is connected
if drone_camera:
    drone = tello.Tello()
    drone.connect()
    print(f"Battery: {drone.get_battery()}%")
    drone.streamon()  # turn on camera
    frame_read = drone.get_frame_read()
    if flying:
        time.sleep(5)
        drone.takeoff()

# If Webcam:
else:
    # Get webcam feed
    cap = cv2.VideoCapture(0)

# loop over frames from the video stream
while True:
    if drone_camera:
        # Drone camera
        orig = frame_read.frame
    else:
        _, orig = cap.read()

    # resize the frame and compute the ratio of the *new* width to
    frame = imutils.resize(orig, height=480, width=640)
    h, w, _ = frame.shape
    ref_x = int(w / 2)
    ref_y = int(h * 0.4)

    # if "c" is press, start detecting
    if cv2.waitKey(1) & 0xFF == ord("c"):
        detecting = not detecting

    if detecting:
        frame = detector.findPose(frame, draw=True)
        lmList = detector.findPosition(frame, draw=True)

        if len(lmList) != 0:
            target = lmList[0]  # Nose landmark
            xoff = int(lmList[0][1] - ref_x)
            yoff = int(ref_y - lmList[0][2])

            ###################################################
            # Calcular distancia entre hombros para
            x_hombro_d = lmList[12][1]
            x_hombro_i = lmList[11][1]

            shoulders_width = x_hombro_i - x_hombro_d

            ###################################################
            proximity = int(w / 3.1)
            keep_distance = proximity

            cv2.circle(frame, (ref_x, ref_y), 15, (250, 150, 0), 1, cv2.LINE_AA)
            cv2.arrowedLine(frame, (ref_x, ref_y), (target[1], target[2]), (250, 150, 0), 5)

            ############################# FACE TRACKING #############################
            # PID CONTROLLERS CALCULATE NEW SPEEDS FOR YAW AND THROTTLE
            roll_speed_value = int(-pid_roll(xoff))
            throttle_speed_value = int(-pid_throttle(yoff))
            pitch_speed_value = int(pid_pitch(shoulders_width - keep_distance))

            i = 0
            if roll_speed_value > 0:
                roll_direction_value = "right"
                cv2.putText(frame, f"RIGHT: {str(roll_speed_value)}", (0, 30 + (i * 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 2555), 2)
            else:
                roll_direction_value = "left"
                cv2.putText(frame, f"LEFT: {str(roll_speed_value)}", (0, 30 + (i * 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 2555), 2)

            i = 1
            if throttle_speed_value > 0:
                throttle_direction_value = "up"
                cv2.putText(frame, f"UP: {str(throttle_speed_value)}", (0, 30 + (i * 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 2555), 2)
            else:
                throttle_direction_value = "down"
                cv2.putText(frame, f"DOWN: {str(throttle_speed_value)}", (0, 30 + (i * 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 2555), 2)

            i = 2
            if pitch_speed_value > 0:
                pitch_direction_value = "forward"
                cv2.putText(frame, f"FORWARD: {str(pitch_speed_value)}", (0, 30 + (i * 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 2555), 2)
            else:
                pitch_direction_value = "back"
                cv2.putText(frame, f"BACK: {str(pitch_speed_value)}", (0, 30 + (i * 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 2555), 2)

            #########################################################################
            ############################# CONTROL DRONE #############################
            if drone_camera:
                drone.move(roll_direction_value, conversion(abs(roll_speed_value)))
                drone.move(throttle_direction_value, conversion(abs(throttle_speed_value)))
                drone.move(pitch_direction_value, conversion(abs(pitch_speed_value)))

    # show the output video OCR visualization
    cv2.imshow("Output", frame)

    # if the 'q' key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        if drone_camera:
            if flying:
                drone.land()
            drone.streamoff()
        else:
            cap.release()
        break


# close any open windows
cv2.destroyAllWindows()



