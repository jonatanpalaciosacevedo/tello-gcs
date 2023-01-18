import imutils
import cv2
import tellopy
import av
import time
import os
import numpy as np
from pynput import keyboard
from modules import pose_module as pm
from simple_pid import PID


def put_text(frame, text, pos):
    cv2.putText(frame, text, (0, 30 + (pos * 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


class TelloController:
    def __init__(self):
        self.keydown = False
        self.fly_speed = None
        self.fly_time = None
        self.ground_speed = None
        self.east_speed = None
        self.north_speed = None
        self.height = None
        self.battery = None
        self.drone = tellopy.Tello()
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
        self.def_speed = {"yaw": 50, "roll": 50, "pitch": 50, "throttle": 80}

    def init_drone(self):
        self.drone.connect()
        self.drone.wait_for_connection(10.0)
        self.drone.start_video()
        self.init_controls()
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, self.flight_data_handler)

    def flight_data_handler(self, event, sender, data):
        self.battery = data.battery_percentage
        self.height = data.height
        self.north_speed = data.north_speed
        self.east_speed = data.east_speed
        self.ground_speed = data.ground_speed
        self.fly_time = data.fly_time
        self.fly_speed = data.fly_speed

    def process_frame(self, raw_frame):
        frame = raw_frame.copy()
        frame = imutils.resize(frame, height=480, width=640)
        h, w, _ = frame.shape

        for axis, command in self.axis_command.items():
            if self.axis_speed[axis] is not None:
                command(self.cmd_axis_speed[axis])

        # Draw on HUD
        self.draw(frame)

        return frame

    def draw(self, frame):
        bat = f"BAT: {int(self.battery)}"

        t = int(self.fly_time)/10
        if t < 60:
            fly_time = f"FLYING: {t} "
        else:
            s = round(((t / 60) % 1) * 60)
            m = int(t/60)
            fly_time = f"FLYING: {m}:{s}"

        put_text(frame, bat, 0)
        put_text(frame, fly_time, 1)
        put_text(frame, f"NORTH SPEED: {self.north_speed}", 2)
        put_text(frame, f"EAST SPEED: {self.east_speed}", 3)
        put_text(frame, f"GROUND SPEED: {self.ground_speed}", 4)
        put_text(frame, f"FLY SPEED: {self.fly_speed}", 5)

    def on_press(self, keyname):
        """
            Handler for keyboard listener
        """
        if self.keydown:
            return
        try:
            self.keydown = True
            keyname = str(keyname).strip('\'')
            if keyname == 'Key.esc':
                # self.tracking = False
                self.drone.land()
                self.drone.quit()
                cv2.destroyAllWindows()
                os._exit(0)
            if keyname in self.controls_keypress:
                self.controls_keypress[keyname]()
        except AttributeError:
            pass

    def on_release(self, keyname):
        """
            Reset on key up from keyboard listener
        """
        self.keydown = False
        keyname = str(keyname).strip('\'')
        if keyname in self.controls_keyrelease:
            key_handler = self.controls_keyrelease[keyname]()

    def set_speed(self, axis, speed):
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
            '0': lambda: self.drone.set_video_encoder_rate(0),
            '1': lambda: self.drone.set_video_encoder_rate(1),
            '2': lambda: self.drone.set_video_encoder_rate(2),
            '3': lambda: self.drone.set_video_encoder_rate(3),
            '4': lambda: self.drone.set_video_encoder_rate(4),
            '5': lambda: self.drone.set_video_encoder_rate(5),
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

        self.controls_keypress = controls_keypress_QWERTY
        self.controls_keyrelease = controls_keyrelease_QWERTY

        self.key_listener = keyboard.Listener(on_press=self.on_press,
                                              on_release=self.on_release)
        self.key_listener.start()


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
        image = CONTROLLER.process_frame(image)
        print(f"{CONTROLLER.ground_speed}")

        cv2.imshow('TELLO', image)

        cv2.waitKey(1)
        if frame.time_base < 1.0 / 60:
            time_base = 1.0 / 60
        else:
            time_base = frame.time_base
        frame_skip = int((time.time() - start_time) / time_base)


if __name__ == '__main__':
    main()
