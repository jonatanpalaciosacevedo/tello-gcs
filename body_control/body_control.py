import time
import datetime
import os
import tellopy
import numpy as np
import av
import cv2
from pynput import keyboard
import argparse
import math

from modules import pose_module as pm
from simple_pid import PID
from multiprocessing import Process, Pipe, sharedctypes
from modules.fps import FPS
from modules.camera_morse import CameraMorse, RollingGraph
import logging
import sys

log = logging.getLogger("TelloMediapipe")


def mediapipe_worker():
    """
    When multiprocessing, this is the init and main loop of the child process
    :return: None
    """
    print("Worker process", os.getpid())
    print("Worker process", os.getpid())
    tello.drone.start_recv_thread()
    tello.init_controls()

    # Start detecting
    tello.pm = pm.PoseDetector()

    # Save video
    while True:
        tello.fps.update()

        frame = np.ctypeslib.as_array(tello.shared_array).copy()
        frame.shape = tello.frame_shape
        frame = tello.process_frame(frame)

        cv2.imshow("Processed", frame)
        cv2.waitKey(1)


def findAngle(p1, p2, p3, lmList):
    """
    Find the angle between 3 landmark points, given a list lmList with landmarks
    :param p1: point 1
    :param p2: point 2
    :param p3: point 3
    :param lmList: List with landmarks with the structure: lmList[int: Landmark][int: x or y pos]
    :return: Angle
    """
    # Get the landmarks
    x1, y1 = lmList[p1][1:]
    x2, y2 = lmList[p2][1:]
    x3, y3 = lmList[p3][1:]

    # Calculate the Angle
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                         math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360

    return angle


def findDistance(x1, y1, x2, y2):
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def quat_to_yaw_deg(qx, qy, qz, qw):
    """
    Calculate Yaw from quaternion
    :param qx: x
    :param qy: y
    :param qz: z
    :param qw: w
    :return: Yaw
    """
    degree = np.pi / 180
    siny = 2 * (qw * qz + qx * qy)
    cosy = 1 - 2 * (qy * qy + qz * qz)
    yaw = int(np.atan2(siny, cosy) / degree)
    return yaw


def main(use_multiprocessing=True, log_level=None):
    """
        Create and run a tello controller :
        1) get the video stream from the tello
        2) wait for keyboard commands to pilot the tello
        3) optionnally, process the video frames to track a body and pilot the tello accordingly.
        If use_multiprocessing is True, the parent process creates a child process ('worker')
        and the workload is shared between the 2 processes.
        The parent process job is to:
        - get the video stream from the tello and displays it in an OpenCV window,
        - write each frame in shared memory at destination of the child,
        each frame replacing the previous one (more efficient than a pipe or a queue),
        - read potential command from the child (currently, only one command:EXIT).
        Commands are transmitted by a Pipe.
        The child process is responsible of all the others tasks:
        - process the frames read in shared memory (openpose, write_hud),
        - if enable, do the tracking (calculate drone commands from position of body),
        - read keyboard commands,
        - transmit commands (from tracking or from keyboard) to the tello, and receive message from the tello.
    """
    global tello

    if use_multiprocessing:
        # Create the pipe for the communication between the 2 processes
        parent_cnx, child_cnx = Pipe()
    else:
        child_cnx = None

    tello = TelloController(use_face_tracking=True, kbd_layout="QWERTY", write_log_data=False,
                            log_level=log_level, child_cnx=child_cnx)

    first_frame = True
    frame_skip = 300

    for frame in tello.container.decode(video=0):
        if 0 < frame_skip:
            frame_skip = frame_skip - 1
            continue
        start_time = time.time()
        if frame.time_base < 1.0 / 60:
            time_base = 1.0 / 60
        else:
            time_base = frame.time_base

        # Convert frame to cv2 image
        frame = cv2.cvtColor(np.array(frame.to_image(), dtype=np.uint8), cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (640, 480))

        if use_multiprocessing:
            if first_frame:
                # Create the shared memory to share the current frame decoded by the parent process
                # and given to the child process for further processing (openpose, write_hud,...)
                frame_as_ctypes = np.ctypeslib.as_ctypes(frame)
                tello.shared_array = sharedctypes.RawArray(frame_as_ctypes._type_, frame_as_ctypes)
                tello.frame_shape = frame.shape
                first_frame = False
                # Launch process child
                p_worker = Process(target=mediapipe_worker())
                p_worker.start()

            # Write the current frame in shared memory
            tello.shared_array[:] = np.ctypeslib.as_ctypes(frame.copy())
            # Check if there is some message from the child
            if parent_cnx.poll():
                msg = parent_cnx.recv()
                if msg == "EXIT":
                    print("MAIN EXIT")
                    p_worker.join()
                    tello.drone.quit()
                    cv2.destroyAllWindows()
                    exit(0)
        else:
            frame = tello.process_frame(frame)

        if not use_multiprocessing:
            tello.fps.update()

        # Display the frame
        cv2.imshow('Tello', frame)

        cv2.waitKey(1)

        frame_skip = int((time.time() - start_time) / time_base)


class TelloController(object):
    """
        TelloController builds keyboard controls on top of TelloPy as well
        as generating images from the video stream and enabling opencv support
    """

    def __init__(self, use_face_tracking=True, kbd_layout="QWERTY", write_log_data=False,
                 media_directory="media", child_cnx=None, log_level=None):

        self.log_level = log_level
        self.debug = log_level is not None
        self.child_cnx = child_cnx
        self.use_multiprocessing = child_cnx is not None
        self.kbd_layout = kbd_layout
        # Flight data
        self.is_flying = False
        self.battery = None
        self.fly_mode = None
        self.throw_fly_timer = 0

        self.tracking_after_takeoff = False
        self.record = False
        self.keydown = False
        self.date_fmt = '%Y-%m-%d_%H%M%S'

        self.drone = tellopy.Tello()
        self.axis_command = {
            "yaw": self.drone.clockwise,
            "roll": self.drone.right,
            "pitch": self.drone.forward,
            "throttle": self.drone.up
        }
        self.axis_speed = {"yaw": 0, "roll": 0, "pitch": 0, "throttle": 0}
        self.cmd_axis_speed = {"yaw": 0, "roll": 0, "pitch": 0, "throttle": 0}
        self.prev_axis_speed = self.axis_speed.copy()
        self.def_speed = {"yaw": 50, "roll": 35, "pitch": 35, "throttle": 80}

        self.write_log_data = write_log_data
        self.reset()
        self.media_directory = media_directory
        if not os.path.isdir(self.media_directory):
            os.makedirs(self.media_directory)

        if self.write_log_data:
            path = 'tello-%s.csv' % datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
            self.log_file = open(path, 'w')
            self.write_header = True

        self.init_drone()
        if not self.use_multiprocessing:
            self.init_controls()

        # container for processing the packets into frames
        self.container = av.open(self.drone.get_video_stream())
        self.vid_stream = self.container.streams.video[0]
        self.out_file = None
        self.out_stream = None
        self.out_name = None
        self.start_time = time.time()

        self.pm = None

        # Setup Openpose
        if not self.use_multiprocessing:
            self.mp = pm.PoseDetector()
        self.use_mediapipe = False

        self.morse = CameraMorse(display=False)
        self.morse.define_command("---", self.delayed_takeoff)
        # self.morse.define_command("...", self.throw_and_go, {'tracking': True})
        self.is_pressed = False

        self.fps = FPS()
        self.exposure = 0

        if self.debug:
            self.graph_pid = RollingGraph(window_name="PID", step_width=2, width=2000, height=500, y_max=200,
                                          colors=[(255, 255, 255), (255, 200, 0), (0, 0, 255), (0, 255, 0)],
                                          thickness=[2, 2, 2, 2], threshold=100, waitKey=False)

        # Logging
        self.log_level = log_level
        if log_level is not None:
            if log_level == "info":
                log_level = logging.INFO
            elif log_level == "debug":
                log_level = logging.DEBUG
            log.setLevel(log_level)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(log_level)
            ch.setFormatter(
                logging.Formatter(fmt='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
                                  datefmt="%H:%M:%S"))
            log.addHandler(ch)

    def set_video_encoder_rate(self, rate):
        self.drone.set_video_encoder_rate(rate)
        self.video_encoder_rate = rate

    def reset(self):
        """
            Reset global variables before a fly
        """
        log.debug("RESET")
        self.ref_pos_x = -1
        self.ref_pos_y = -1
        self.ref_pos_z = -1
        self.pos_x = -1
        self.pos_y = -1
        self.pos_z = -1
        self.yaw = 0
        self.tracking = False
        self.keep_distance = None
        self.palm_landing = False
        self.palm_landing_approach = False
        self.yaw_to_consume = 0
        self.timestamp_keep_distance = time.time()
        self.wait_before_tracking = None
        self.timestamp_take_picture = None
        self.throw_ongoing = False
        self.scheduled_takeoff = None
        # When in trackin mode, but no body is detected in current frame,
        # we make the drone rotate in the hope to find some body
        # The rotation is done in the same direction as the last rotation done
        self.body_in_prev_frame = False
        self.timestamp_no_body = time.time()
        self.last_rotation_is_cw = True

    def init_drone(self):
        # Connect to the drone, start streaming and subscribe to events
        self.drone.connect()
        self.drone.wait_for_connection(10.0)
        self.set_video_encoder_rate(2)
        self.drone.start_video()
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA,
                             self.flight_data_handler)
        # self.drone.subscribe(self.drone.EVENT_LOG,
        #                      self.log_data_handler)
        self.drone.subscribe(self.drone.EVENT_FILE_RECEIVED,
                             self.handle_flight_received)

    def on_press(self, keyname):
        """
            Handler for keyboard listener
        """
        if self.keydown:
            return
        try:
            self.keydown = True
            keyname = str(keyname).strip('\'')
            log.info('KEY PRESS ' + keyname)
            if keyname == 'Key.esc':
                self.toggle_tracking(False)
                # self.tracking = False
                self.drone.land()
                self.drone.quit()
                if self.child_cnx:
                    # Tell to the parent process that it's time to exit
                    self.child_cnx.send("EXIT")
                cv2.destroyAllWindows()
                os._exit(0)
            if keyname in self.controls_keypress:
                self.controls_keypress[keyname]()
        except AttributeError:
            log.debug(f'special key {keyname} pressed')

    def on_release(self, keyname):
        """
            Reset on key up from keyboard listener
        """
        self.keydown = False
        keyname = str(keyname).strip('\'')
        log.info('KEY RELEASE ' + keyname)
        if keyname in self.controls_keyrelease:
            key_handler = self.controls_keyrelease[keyname]()

    def set_speed(self, axis, speed):
        log.info(f"set speed {axis} {speed}")
        self.cmd_axis_speed[axis] = speed

    def init_controls(self):
        """
            Define keys and add listener
        """

        controls_keypress_QWERTY = {
            'w': lambda: self.set_speed("pitch", self.def_speed["pitch"]),
            's': lambda: self.set_speed("pitch", -self.def_speed["pitch"]),
            'a': lambda: self.set_speed("roll", -self.def_speed["roll"]),
            'd': lambda: self.set_speed("roll", self.def_speed["roll"]),
            'q': lambda: self.set_speed("yaw", -self.def_speed["yaw"]),
            'e': lambda: self.set_speed("yaw", self.def_speed["yaw"]),
            'i': lambda: self.drone.flip_forward(),
            'k': lambda: self.drone.flip_back(),
            'j': lambda: self.drone.flip_left(),
            'l': lambda: self.drone.flip_right(),
            'Key.left': lambda: self.set_speed("yaw", -1.5 * self.def_speed["yaw"]),
            'Key.right': lambda: self.set_speed("yaw", 1.5 * self.def_speed["yaw"]),
            'Key.up': lambda: self.set_speed("throttle", self.def_speed["throttle"]),
            'Key.down': lambda: self.set_speed("throttle", -self.def_speed["throttle"]),
            'Key.tab': lambda: self.drone.takeoff(),
            'Key.backspace': lambda: self.drone.land(),
            'p': lambda: self.palm_land(),
            't': lambda: self.toggle_tracking(),
            'o': lambda: self.toggle_mediapipe(),
            'Key.enter': lambda: self.take_picture(),
            'c': lambda: self.clockwise_degrees(360),
            '0': lambda: self.drone.set_video_encoder_rate(0),
            '1': lambda: self.drone.set_video_encoder_rate(1),
            '2': lambda: self.drone.set_video_encoder_rate(2),
            '3': lambda: self.drone.set_video_encoder_rate(3),
            '4': lambda: self.drone.set_video_encoder_rate(4),
            '5': lambda: self.drone.set_video_encoder_rate(5),

            '7': lambda: self.set_exposure(-1),
            '8': lambda: self.set_exposure(0),
            '9': lambda: self.set_exposure(1)
        }

        controls_keyrelease_QWERTY = {
            'w': lambda: self.set_speed("pitch", 0),
            's': lambda: self.set_speed("pitch", 0),
            'a': lambda: self.set_speed("roll", 0),
            'd': lambda: self.set_speed("roll", 0),
            'q': lambda: self.set_speed("yaw", 0),
            'e': lambda: self.set_speed("yaw", 0),
            'Key.left': lambda: self.set_speed("yaw", 0),
            'Key.right': lambda: self.set_speed("yaw", 0),
            'Key.up': lambda: self.set_speed("throttle", 0),
            'Key.down': lambda: self.set_speed("throttle", 0)
        }

        controls_keypress_AZERTY = {
            'z': lambda: self.set_speed("pitch", self.def_speed["pitch"]),
            's': lambda: self.set_speed("pitch", -self.def_speed["pitch"]),
            'q': lambda: self.set_speed("roll", -self.def_speed["roll"]),
            'd': lambda: self.set_speed("roll", self.def_speed["roll"]),
            'a': lambda: self.set_speed("yaw", -self.def_speed["yaw"]),
            'e': lambda: self.set_speed("yaw", self.def_speed["yaw"]),
            'i': lambda: self.drone.flip_forward(),
            'k': lambda: self.drone.flip_back(),
            'j': lambda: self.drone.flip_left(),
            'l': lambda: self.drone.flip_right(),
            'Key.left': lambda: self.set_speed("yaw", -1.5 * self.def_speed["yaw"]),
            'Key.right': lambda: self.set_speed("yaw", 1.5 * self.def_speed["yaw"]),
            'Key.up': lambda: self.set_speed("throttle", self.def_speed["throttle"]),
            'Key.down': lambda: self.set_speed("throttle", -self.def_speed["throttle"]),
            'Key.tab': lambda: self.drone.takeoff(),
            'Key.backspace': lambda: self.drone.land(),
            'p': lambda: self.palm_land(),
            't': lambda: self.toggle_tracking(),
            'o': lambda: self.toggle_mediapipe(),
            'Key.enter': lambda: self.take_picture(),
            'c': lambda: self.clockwise_degrees(360),
            '0': lambda: self.drone.set_video_encoder_rate(0),
            '1': lambda: self.drone.set_video_encoder_rate(1),
            '2': lambda: self.drone.set_video_encoder_rate(2),
            '3': lambda: self.drone.set_video_encoder_rate(3),
            '4': lambda: self.drone.set_video_encoder_rate(4),
            '5': lambda: self.drone.set_video_encoder_rate(5),

            '7': lambda: self.set_exposure(-1),
            '8': lambda: self.set_exposure(0),
            '9': lambda: self.set_exposure(1)
        }

        controls_keyrelease_AZERTY = {
            'z': lambda: self.set_speed("pitch", 0),
            's': lambda: self.set_speed("pitch", 0),
            'q': lambda: self.set_speed("roll", 0),
            'd': lambda: self.set_speed("roll", 0),
            'a': lambda: self.set_speed("yaw", 0),
            'e': lambda: self.set_speed("yaw", 0),
            'Key.left': lambda: self.set_speed("yaw", 0),
            'Key.right': lambda: self.set_speed("yaw", 0),
            'Key.up': lambda: self.set_speed("throttle", 0),
            'Key.down': lambda: self.set_speed("throttle", 0)
        }

        if self.kbd_layout == "AZERTY":
            self.controls_keypress = controls_keypress_AZERTY
            self.controls_keyrelease = controls_keyrelease_AZERTY
        else:
            self.controls_keypress = controls_keypress_QWERTY
            self.controls_keyrelease = controls_keyrelease_QWERTY
        self.key_listener = keyboard.Listener(on_press=self.on_press,
                                              on_release=self.on_release)
        self.key_listener.start()

    def check_pose(self, lmList):
        if len(lmList) != 0:
            # Check if we detect a pose in the body detected by Openpose
            """
            left_arm_angle = detector.findAngle(img, 11, 13, 21)  # Brazo izquierdo
            right_arm_angle = detector.findAngle(img, 22, 14, 12)  # Brazo derecho
            left_arm_angle2 = detector.findAngle(img, 13, 11, 23)  # Brazo izquierdo
            right_arm_angle2 = detector.findAngle(img, 24, 12, 14)  # Brazo derecho
            """
            # Arms control roll
            left_arm_angle = findAngle(11, 13, 21, lmList)
            left_arm_angle2 = findAngle(13, 11, 23, lmList)

            right_arm_angle = findAngle(22, 14, 12, lmList)
            right_arm_angle2 = findAngle(24, 12, 14, lmList)

            move_left = False
            move_right = False
            take_pic = False
            land_drone = False

            # Left arm up at around 90+ degrees with body, the drone moves to its right
            if 110 > left_arm_angle2 > 80 and 180 > left_arm_angle > 100:
                move_right = True

            # Right arm up at around 90+ degrees with body, the drone moves to its left
            if 110 > right_arm_angle2 > 80 and 180 > right_arm_angle > 100:
                move_left = True

            # Arms up and elbows folded
            if 110 > right_arm_angle2 > 80 and right_arm_angle < 50 and 110 > left_arm_angle2 > 80 and left_arm_angle < 50:
                take_pic = True

            # Wrists cross over head
            if left_arm_angle2 > 135 and 180 > left_arm_angle > 100 and right_arm_angle2 > 135 and 180 > right_arm_angle > 100:
                land_drone = True

            # Calculates shoulder distance to keep distance
            right_shoulder_x = lmList[12][1]
            left_shoulder_x = lmList[11][1]
            shoulders_width = left_shoulder_x - right_shoulder_x
            self.shoulders_width = shoulders_width

            if move_left:
                return "MOVING_LEFT"

            if move_right:
                return "MOVING_RIGHT"

            if take_pic:
                return "TAKING_PICTURE"

            if land_drone:
                return "LANDING_DRONE"

        else:
            return None

    def process_frame(self, raw_frame):
        """
            Analyze the frame and return the frame with information (HUD, openpose skeleton) drawn on it
        """
        frame = raw_frame.copy()
        h, w, _ = frame.shape

        # Is there a scheduled takeoff ?
        if self.scheduled_takeoff and time.time() > self.scheduled_takeoff:
            self.scheduled_takeoff = None
            self.drone.takeoff()

        self.axis_speed = self.cmd_axis_speed.copy()

        # If we are on the point to take a picture, the tracking is temporarily deactivated (2s)
        if self.timestamp_take_picture:
            if time.time() - self.timestamp_take_picture > 2:
                self.timestamp_take_picture = None
                self.drone.take_picture()

        else:
            # If we are doing a 360, where are we in our 360 ?
            if self.yaw_to_consume > 0:
                consumed = self.yaw - self.prev_yaw
                self.prev_yaw = self.yaw
                if consumed < 0:
                    consumed += 360
                self.yaw_consumed += consumed
                if self.yaw_consumed > self.yaw_to_consume:
                    self.yaw_to_consume = 0
                    self.axis_speed["yaw"] = 0
                else:
                    self.axis_speed["yaw"] = self.def_speed["yaw"]

            # We are not flying, we check a potential morse code
            if not self.is_flying:
                pressing, detected = self.morse.eval(frame)
                self.is_pressed = pressing

            # Call to mediapipe detection
            if self.use_mediapipe:
                frame = self.mp.findPose(frame, draw=True)
                lmList = self.mp.findPosition(frame, draw=True)

                target = None

                # Our target is the person whose index is 0 in pose_kps
                self.pose = None
                if len(lmList) != 0:
                    # We found a body, so we can cancel the exploring 360
                    self.yaw_to_consume = 0

                    # Do we recognize a predefined pose ?
                    self.pose = self.check_pose(lmList)

                    if self.pose:
                        # We trigger the associated action
                        log.info(f"pose detected : {self.pose}")
                        if self.pose == "MOVING_RIGHT":
                            log.info("GOING LEFT from pose")
                            self.axis_speed["roll"] = self.def_speed["roll"]

                        elif self.pose == "MOVING_LEFT":
                            log.info("GOING RIGHT from pose")
                            self.axis_speed["roll"] = -self.def_speed["roll"]

                        elif self.pose == "TAKING_PICTURE":
                            # Take a picture in 1 second
                            if self.timestamp_take_picture is None:
                                log.info("Take a picture in 1 second")
                                self.timestamp_take_picture = time.time()

                        elif self.pose == "LANDING_DRONE":
                            if not self.palm_landing:
                                log.info("LANDING on pose")
                                # Landing
                                self.toggle_tracking(tracking=False)
                                self.drone.land()

                    target = lmList[0]  # Nose landmark

                ref_x = int(w / 2)
                ref_y = int(h * 0.25)

                if self.tracking:
                    if target:
                        self.body_in_prev_frame = True
                        # Calcular face tracking
                        xoff = int(lmList[0][1] - ref_x)
                        yoff = int(ref_y - lmList[0][2])
                        cv2.circle(frame, (ref_x, ref_y), 15, (250, 150, 0), 1, cv2.LINE_AA)
                        cv2.arrowedLine(frame, (ref_x, ref_y), (target[1], target[2]), (250, 150, 0), 5)

                        proximity = int(w / 3.1)
                        self.keep_distance = proximity

                        # PID CONTROLLERS CALCULATE NEW SPEEDS FOR YAW AND THROTTLE
                        self.axis_speed["yaw"] = int(-self.pid_yaw(xoff))
                        log.debug(f"xoff: {xoff} - speed_yaw: {self.axis_speed['yaw']}")
                        self.last_rotation_is_cw = self.axis_speed["yaw"] > 0

                        self.axis_speed["throttle"] = int(-self.pid_throttle(yoff))
                        log.debug(f"yoff: {yoff} - speed_throttle: {self.axis_speed['throttle']}")

                        self.axis_speed["pitch"] = int(self.pid_pitch(self.shoulders_width - self.keep_distance))
                        log.debug(f"speed_pitch: {self.axis_speed['pitch']}")

                    else:  # Tracking but no body detected
                        if self.body_in_prev_frame:
                            self.body_in_prev_frame = False
                            self.axis_speed["throttle"] = self.prev_axis_speed["throttle"]
                            self.axis_speed["yaw"] = self.prev_axis_speed["yaw"]
                        else:
                            if time.time() - self.timestamp_no_body < 1:
                                print("NO BODY SINCE < 1", self.axis_speed, self.prev_axis_speed)
                                self.axis_speed["throttle"] = self.prev_axis_speed["throttle"]
                                self.axis_speed["yaw"] = self.prev_axis_speed["yaw"]
                            else:
                                log.debug("NO BODY detected for 1s -> rotate")
                                self.axis_speed["yaw"] = self.def_speed["yaw"] * (1 if self.last_rotation_is_cw else -1)

        # Send axis commands to the drone
        for axis, command in self.axis_command.items():
            if self.axis_speed[axis] is not None and self.axis_speed[axis] != self.prev_axis_speed[axis]:
                log.debug(f"COMMAND {axis} : {self.axis_speed[axis]}")
                command(self.axis_speed[axis])
                self.prev_axis_speed[axis] = self.axis_speed[axis]
            else:
                # This line is necessary to display current values in 'self.^'
                self.axis_speed[axis] = self.prev_axis_speed[axis]

        # Write the HUD on the frame
        frame = self.write_hud(frame)

        return frame

    def write_hud(self, frame):
        """
            Draw drone info on frame
        """

        class HUD:
            def __init__(self, def_color=(255, 170, 0)):
                self.def_color = def_color
                self.infos = []

            def add(self, info, color=None):
                if color is None: color = self.def_color
                self.infos.append((info, color))

            def draw(self, frame):
                i = 0
                for (info, color) in self.infos:
                    cv2.putText(frame, info, (0, 30 + (i * 30)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, color, 2)  # lineType=30)
                    i += 1

        hud = HUD()

        if self.debug: hud.add(datetime.datetime.now().strftime('%H:%M:%S'))
        hud.add(f"FPS {self.fps.get():.2f}")
        if self.debug: hud.add(f"VR {self.video_encoder_rate}")

        hud.add(f"BAT {self.battery}")
        if self.is_flying:
            hud.add("FLYING", (0, 255, 0))
        else:
            hud.add("NOT FLYING", (0, 0, 255))
        hud.add(f"TRACKING {'ON' if self.tracking else 'OFF'}", (0, 255, 0) if self.tracking else (0, 0, 255))
        # hud.add(f"EXPO {self.exposure}")

        if self.axis_speed['yaw'] > 0:
            hud.add(f"CW {self.axis_speed['yaw']}", (0, 255, 0))
        elif self.axis_speed['yaw'] < 0:
            hud.add(f"CCW {-self.axis_speed['yaw']}", (0, 0, 255))
        else:
            hud.add(f"CW 0")
        if self.axis_speed['roll'] > 0:
            hud.add(f"RIGHT {self.axis_speed['roll']}", (0, 255, 0))
        elif self.axis_speed['roll'] < 0:
            hud.add(f"LEFT {-self.axis_speed['roll']}", (0, 0, 255))
        else:
            hud.add(f"RIGHT 0")
        if self.axis_speed['pitch'] > 0:
            hud.add(f"FORWARD {self.axis_speed['pitch']}", (0, 255, 0))
        elif self.axis_speed['pitch'] < 0:
            hud.add(f"BACKWARD {-self.axis_speed['pitch']}", (0, 0, 255))
        else:
            hud.add(f"FORWARD 0")
        if self.axis_speed['throttle'] > 0:
            hud.add(f"UP {self.axis_speed['throttle']}", (0, 255, 0))
        elif self.axis_speed['throttle'] < 0:
            hud.add(f"DOWN {-self.axis_speed['throttle']}", (0, 0, 255))
        else:
            hud.add(f"UP 0")

        if self.use_mediapipe: hud.add(f"POSE: {self.pose}", (0, 255, 0) if self.pose else (255, 170, 0))
        if self.keep_distance:
            hud.add(f"Target distance: {self.keep_distance} - curr: {self.shoulders_width}", (0, 255, 0))
            # if self.shoulders_width: self.graph_distance.new_iter([self.shoulders_width])
        if self.timestamp_take_picture: hud.add("Taking a picture", (0, 255, 0))
        if self.palm_landing:
            hud.add("Palm landing...", (0, 255, 0))
        if self.palm_landing_approach:
            hud.add("In approach for palm landing...", (0, 255, 0))
        if self.tracking and not self.body_in_prev_frame and time.time() - self.timestamp_no_body > 0.5:
            hud.add("Searching...", (0, 255, 0))
        if self.throw_ongoing:
            hud.add("Throw ongoing...", (0, 255, 0))
        if self.scheduled_takeoff:
            seconds_left = int(self.scheduled_takeoff - time.time())
            hud.add(f"Takeoff in {seconds_left}s")

        hud.draw(frame)
        return frame

    def take_picture(self):
        """
            Tell drone to take picture, image sent to file handler
        """
        self.drone.take_picture()

    def set_exposure(self, expo):
        """
            Change exposure of drone camera
        """
        if expo == 0:
            self.exposure = 0
        elif expo == 1:
            self.exposure = min(9, self.exposure + 1)
        elif expo == -1:
            self.exposure = max(-9, self.exposure - 1)
        self.drone.set_exposure(self.exposure)
        log.info(f"EXPOSURE {self.exposure}")

    def palm_land(self):
        """
            Tell drone to land
        """
        self.palm_landing = True
        self.drone.palm_land()

    def throw_and_go(self, tracking=False):
        """
            Tell drone to start a 'throw and go'
        """
        self.drone.throw_and_go()
        self.tracking_after_takeoff = tracking

    def delayed_takeoff(self, delay=5):
        self.scheduled_takeoff = time.time() + delay
        self.tracking_after_takeoff = True

    def clockwise_degrees(self, degrees):
        self.yaw_to_consume = degrees
        self.yaw_consumed = 0
        self.prev_yaw = self.yaw

    def toggle_mediapipe(self):
        self.use_mediapipe = not self.use_mediapipe
        if not self.use_mediapipe:
            # Desactivate tracking
            self.toggle_tracking(tracking=False)
        log.info('MEDIAPIPE ' + ("ON" if self.use_mediapipe else "OFF"))

    def toggle_tracking(self, tracking=None):
        """
            If tracking is None, toggle value of self.tracking
            Else self.tracking take the same value as tracking
        """

        if tracking is None:
            self.tracking = not self.tracking
        else:
            self.tracking = tracking
        if self.tracking:
            log.info("ACTIVATE TRACKING")
            # Needs openpose
            self.use_mediapipe = True
            # Start an explarotary 360
            # self.clockwise_degrees(360)
            # Init a PID controller for the yaw
            self.pid_yaw = PID(0.25, 0, 0, setpoint=0, output_limits=(-100, 100))
            self.pid_throttle = PID(0.4, 0, 0, setpoint=0, output_limits=(-80, 100))
            self.pid_pitch = PID(0.4, 0, 0, setpoint=0, output_limits=(-50, 50))

        else:
            self.axis_speed = {"yaw": 0, "roll": 0, "pitch": 0, "throttle": 0}
            self.keep_distance = None
        return

    def flight_data_handler(self, event, sender, data):
        """
            Listener to flight data from the drone.
        """
        self.battery = data.battery_percentage
        self.fly_mode = data.fly_mode
        self.throw_fly_timer = data.throw_fly_timer
        self.throw_ongoing = data.throw_fly_timer > 0
        if self.is_flying != data.em_sky:
            self.is_flying = data.em_sky
            log.debug(f"FLYING : {self.is_flying}")
            if not self.is_flying:
                self.reset()
            else:
                if self.tracking_after_takeoff:
                    log.info("Tracking on after takeoff")
                    self.toggle_tracking(True)

        log.debug(f"MODE: {self.fly_mode} - Throw fly timer: {self.throw_fly_timer}")

    def log_data_handler(self, event, sender, data):
        """
            Listener to log data from the drone. CURRENTLY NOT WORKING
        """
        pos_x = -data.mvo.pos_x
        pos_y = -data.mvo.pos_y
        pos_z = -data.mvo.pos_z
        if abs(pos_x) + abs(pos_y) + abs(pos_z) > 0.07:
            if self.ref_pos_x == -1:  # First time we have meaningful values, we store them as reference
                self.ref_pos_x = pos_x
                self.ref_pos_y = pos_y
                self.ref_pos_z = pos_z
            else:
                self.pos_x = pos_x - self.ref_pos_x
                self.pos_y = pos_y - self.ref_pos_y
                self.pos_z = pos_z - self.ref_pos_z

        qx = data.imu.q1
        qy = data.imu.q2
        qz = data.imu.q3
        qw = data.imu.q0
        self.yaw = quat_to_yaw_deg(qx, qy, qz, qw)

        if self.write_log_data:
            if self.write_header:
                self.log_file.write('%s\n' % data.format_cvs_header())
                self.write_header = False
            self.log_file.write('%s\n' % data.format_cvs())

    def handle_flight_received(self, event, sender, data):
        """
            Create a file in local directory to receive image from the drone
        """
        path = f'{self.media_directory}/tello-{datetime.datetime.now().strftime(self.date_fmt)}.jpg'
        with open(path, 'wb') as out_file:
            out_file.write(data)
        log.info('Saved photo to %s' % path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--log_level", help="select a log level (info, debug)")
    ap.add_argument("-2", "--multiprocess", action='store_true',
                    help="use 2 processes to share the workload (instead of 1)")
    args = ap.parse_args()

    main(use_multiprocessing=args.multiprocess, log_level=args.log_level)
