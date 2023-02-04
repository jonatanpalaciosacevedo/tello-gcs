import imutils
import cv2
import tellopy
import av
import time
import numpy as np
from modules import pose_module as pm
from simple_pid import PID


"""
Este codigo solo sigue la cara de una persona haciendo movimientos en los ejes "Z" e "Y" del dron. 
Es decir movimientos de Roll y Throttle

"""


def put_text(frame, text, pos):
    cv2.putText(frame, text, (0, 30 + (pos * 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


class TelloController:
    def __init__(self):

        # Drone state
        self.drone = tellopy.Tello()
        self.flying = False
        self.detecting = False

        # Detection values
        self.detector = pm.PoseDetector()
        self.pid_yaw = PID(0.25, 0, 0, setpoint=0, output_limits=(-100, 100))
        self.pid_throttle = PID(0.4, 0, 0, setpoint=0, output_limits=(-80, 100))
        self.pid_pitch = PID(0.4, 0, 0, setpoint=0, output_limits=(-50, 50))
        self.pid_roll = PID(0.2, 0, 0.1, setpoint=0, output_limits=(-100, 100))

        # Init drone
        self.battery = None
        self.north = None
        self.east = None

        self.init_drone()
        self.axis_command = {
            "yaw": self.drone.clockwise,
            "roll": self.drone.right,
            "pitch": self.drone.forward,
            "throttle": self.drone.up
        }
        self.axis_speed = {"yaw": 0, "roll": 0, "pitch": 0, "throttle": 0}
        self.cmd_axis_speed = {"yaw": 0, "roll": 0, "pitch": 0, "throttle": 0}
        self.prev_axis_speed = self.axis_speed.copy()

    def init_drone(self):
        # Connect to the drone, start video
        self.drone.connect()
        self.drone.wait_for_connection(20.0)
        self.drone.start_video()
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA,
                             self.flight_data_handler)

    def flight_data_handler(self, event, sender, data):
        self.battery = data.battery_percentage
        self.north = data.north_speed
        self.east = data.east_speed

    def speed_controller(self, raw_frame):
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.quit()

        if cv2.waitKey(1) & 0xFF == ord("l"):
            self.drone.land()

        if cv2.waitKey(1) & 0xFF == ord("t"):
            self.flying = not self.flying
            if self.flying:
                self.drone.takeoff()
            else:
                self.drone.land()

        if cv2.waitKey(1) & 0xFF == ord("d"):
            self.detecting = not self.detecting
            print(f"Detecting = {self.detecting}")

        frame = raw_frame.copy()
        frame = imutils.resize(frame, height=480, width=640)
        h, w, _ = frame.shape

        ref_x = int(w / 2)
        ref_y = int(h * 0.4)

        if self.detecting:
            frame = self.detector.findPose(frame, draw=True)
            lmList = self.detector.findPosition(frame, draw=True)

            if len(lmList) != 0:
                target = lmList[0]  # Nose landmark

                if target:
                    # Calculate the xoff and yoff values for the PID controller (ROLL)
                    xoff = int(lmList[0][1] - ref_x)
                    yoff = int(ref_y - lmList[0][2])

                    # Calculate distance between shoulders to controll PITCH
                    right_shoulder_x = lmList[12][1]
                    left_shoulder_x = lmList[11][1]
                    shoulders_width = left_shoulder_x - right_shoulder_x
                    self.shoulders_width = shoulders_width
                    proximity = int(w / 3.1)
                    self.keep_distance = proximity

                    # Draw arrow to the nose to show the distance correction needed
                    cv2.circle(frame, (ref_x, ref_y), 15, (250, 150, 0), 1, cv2.LINE_AA)
                    cv2.arrowedLine(frame, (ref_x, ref_y), (target[1], target[2]), (250, 150, 0), 5)

                    # Face tracking PID controller
                    self.axis_speed["roll"] = int(-self.pid_roll(xoff))
                    self.axis_speed["throttle"] = int(-self.pid_throttle(yoff))
                    self.axis_speed["pitch"] = int(self.pid_pitch(self.shoulders_width - self.keep_distance))

        # Send commands to the drone
        if self.flying:
            for axis, command in self.axis_command.items():
                if self.axis_speed[axis] is not None and self.axis_speed[axis] != self.prev_axis_speed[axis]:
                    command(self.axis_speed[axis])
                    self.prev_axis_speed[axis] = self.axis_speed[axis]
                else:
                    # This line is necessary to display current values in 'self.^'
                    self.axis_speed[axis] = self.prev_axis_speed[axis]

        # Draw on HUD
        self.draw(frame)

        return frame

    def draw(self, frame):
        bat = f"BAT: {int(self.battery)}"
        if self.axis_speed["throttle"] > 0:
            thr = f"UP: {int(self.axis_speed['throttle'])}"
        else:
            thr = f"DOWN: {int(self.axis_speed['throttle'])}"
        if self.axis_speed["roll"] > 0:
            roll = f"RIGHT: {int(self.axis_speed['roll'])}"
        else:
            roll = f"LEFT: {int(self.axis_speed['roll'])}"

        put_text(frame, bat, 0)
        put_text(frame, thr, 1)
        put_text(frame, roll, 2)

    def quit(self):
        if self.flying:
            self.drone.land()
            self.flying = False

        self.drone.quit()
        cv2.destroyAllWindows()


def main():
    CONTROLLER = TelloController()

    frame_skip = 300
    container = av.open(CONTROLLER.drone.get_video_stream())

    for frame in container.decode(video=0):
        if 0 < frame_skip:
            frame_skip = frame_skip - 1
            continue
        start_time = time.time()
        image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
        image = CONTROLLER.speed_controller(image)

        cv2.imshow('TELLO', image)

        cv2.waitKey(1)
        if frame.time_base < 1.0 / 60:
            time_base = 1.0 / 60
        else:
            time_base = frame.time_base
        frame_skip = int((time.time() - start_time) / time_base)


if __name__ == '__main__':
    main()
