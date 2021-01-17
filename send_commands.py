#!/usr/bin/env python2
from __future__ import print_function

import time
import numpy as np
import threading
import robobo
import cv2
import sys
import signal
import prey_predator
from pynput import keyboard

PRESSED = ""
ACTIONS = {
    "up": (40, 40),
    "left": (0, 75),
    "right": (75, 0),
    "down": (-30, -30)
}

def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def keyboard_action(key):
    global PRESSED
    if key == keyboard.Key.enter:
        PRESSED = "exit"
    elif key == keyboard.Key.up:
        PRESSED = "up"
    elif key == keyboard.Key.right:
        PRESSED = "right"
    elif key == keyboard.Key.left:
        PRESSED = "left"
    elif key == keyboard.Key.down:
        PRESSED = "down"

def move_robot(rob):
    global PRESSED
    while PRESSED != "exit":
        if PRESSED == "" or PRESSED == "exit":
            continue
        print("Executing " + PRESSED)
        # print("robobo is at {}".format(rob.position()))
        rob.move(ACTIONS[PRESSED][0], ACTIONS[PRESSED][1], 1000)
        # IR reading
        # for i in range(1000000):
        print("ROB Irs: {}".format(np.log(np.array(rob.read_irs())) / 10))
        PRESSED = ""
        # time.sleep(0.5)

def main():
    signal.signal(signal.SIGINT, terminate_program)

    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)

    rob.play_simulation()
    l1 = keyboard.Listener(on_press=lambda key: keyboard_action(key))
    l1.start()
    thread = threading.Thread(target=move_robot, args=[rob,])
    thread.start()
        # Following code moves the robot
    # while
    #         print("robobo is at {}".format(rob.position()))
    #         rob.move(50, 5, 2000)
    # # IR reading
    # for i in range(1000000):
    #     print("ROB Irs: {}".format(np.log(np.array(rob.read_irs()))/10))
    #     time.sleep(0.1)

    print("robobo is at {}".format(rob.position()))
    rob.sleep(1)

    # Following code moves the phone stand
    # rob.set_phone_pan(343, 100)
    # rob.set_phone_tilt(109, 100)
    # time.sleep(1)
    rob.set_phone_pan(11, 100)
    # rob.set_phone_tilt(26, 100)

    # Following code makes the robot talk and be emotional
    # rob.set_emotion('happy')
    # rob.talk('Hi, my name is Robobo')
    # rob.sleep(1)
    # rob.set_emotion('sad')
    #
    # # Following code gets an image from the camera
    # image = rob.get_image_front()
    # cv2.imwrite("test_pictures.png",image)
    #
    # time.sleep(0.1)

    # # IR reading
    # for i in range(1000000):
    #     print("ROB Irs: {}".format(np.log(np.array(rob.read_irs()))/10))
    #     time.sleep(0.1)

    # pause the simulation and read the collected food
    rob.pause_simulation()
    
    # Stopping the simualtion resets the environment
    rob.stop_world()


if __name__ == "__main__":
    main()
