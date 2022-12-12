import imutils
import cv2
import tellopy
import av
import time
import numpy as np
from modules import pose_module as pm
from simple_pid import PID


def conversion(old_value):
    # OldRange = (OldMax - OldMin)
    # NewRange = (NewMax - NewMin)
    # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin

    old_max = 100
    old_min = -100
    new_max = 30
    new_min = 20
    old_range = old_max - old_min
    new_range = new_max - new_min

    new_value = ((old_value - old_min) * new_range / old_range) + new_min

    return new_value


class TelloController:
    def __init__(self):

        # Drone state
        self.drone = tellopy.Tello()
        self.flying = False
        self.detecting = False
        self.rotate = False
        self.body_in_prev_frame = False

        # Detection values
        self.detector = pm.PoseDetector()
        self.pid_yaw = PID(0.25, 0, 0, setpoint=0, output_limits=(-100, 100))
        self.pid_throttle = PID(0.4, 0, 0, setpoint=0, output_limits=(-80, 100))
        self.pid_pitch = PID(0.4, 0, 0, setpoint=0, output_limits=(-50, 50))
        self.pid_roll = PID(0.35, 0, 0, setpoint=0, output_limits=(-80, 80))
        self.roll_speed_value = None
        self.throttle_speed_value = None

        # Init drone
        self.battery = None
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

        # Other variables
        self.target = None

    def init_drone(self):
        # Connect to the drone, start video
        self.drone.connect()
        self.drone.wait_for_connection(20.0)
        self.drone.start_video()
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA,
                             self.flight_data_handler)
        if self.battery:
            print(f"Battery: {self.battery}%")

    def flight_data_handler(self, event, sender, data):
        """
            Listener to flight data from the drone.
        """
        self.battery = data.battery_percentage

    def spin_360(self):
        self.axis_speed["right"] = 0
        self.axis_speed["left"] = 0
        self.axis_speed["up"] = 0
        self.axis_speed["down"] = 0
        self.drone.rotate_clockwise(1)

    def speed_controller(self, raw_frame):

        if self.rotate:
            self.spin_360()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.quit()

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
                    self.body_in_prev_frame = True
                    self.rotate = False
                    xoff = int(lmList[0][1] - ref_x)
                    yoff = int(ref_y - lmList[0][2])

                    cv2.circle(frame, (ref_x, ref_y), 15, (250, 150, 0), 1, cv2.LINE_AA)
                    cv2.arrowedLine(frame, (ref_x, ref_y), (target[1], target[2]), (250, 150, 0), 5)

                    ############################# FACE TRACKING #############################
                    # PID CONTROLLERS CALCULATE NEW SPEEDS FOR YAW AND THROTTLE
                    self.axis_speed["roll"] = int(-self.pid_roll(xoff))
                    self.axis_speed["throttle"] = int(-self.pid_throttle(yoff))

        if self.flying:
            # cv2.putText(frame, f"RIGHT: {str(roll_speed_value)}", (0, 30 + (i * 30)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 2555), 2)
            for axis, command in self.axis_command.items():
                if self.axis_speed[axis] is not None and self.axis_speed[axis] != self.prev_axis_speed[axis]:
                    command(self.axis_speed[axis])
                    self.prev_axis_speed[axis] = self.axis_speed[axis]
                else:
                    # This line is necessary to display current values in 'self.^'
                    self.axis_speed[axis] = self.prev_axis_speed[axis]

        return frame

    def quit(self):
        if self.flying:
            self.drone.land()

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
        #cv2.putText(frame, f"BAT: {CONTROLLER.battery}", (0, 30 + (0 * 30)),
         #           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 2555), 2)

        cv2.imshow('TELLO', image)

        cv2.waitKey(1)
        if frame.time_base < 1.0 / 60:
            time_base = 1.0 / 60
        else:
            time_base = frame.time_base
        frame_skip = int((time.time() - start_time) / time_base)


if __name__ == '__main__':
    print("Press 't' to takeoff")
    print("Press 'd' to start detecting")
    print("Press 'q' to start stop")
    main()
