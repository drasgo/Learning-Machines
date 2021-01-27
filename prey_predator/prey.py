import random
import threading

import torch

from utils import get_ir_signal


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
        regularly for the stopped() condition."""

    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

LEVEL = {
    1: {},
    2: {}
}

class Prey(StoppableThread):
    def __init__(self, robot, model, device, level=1):
        super(Prey, self).__init__()
        self.model = model
        self._robot = robot
        # default level is 2 -> medium
        self._level = level
        self.device = device

    def run(self):
        prev_out = -1
        counter = 0
        while not self.stopped():
            ir = get_ir_signal(self._robot, self.device)
            # Net output
            outputs = self.model(ir)
            _, output = torch.max(outputs.data, 1)
            output = output.item()

            # Check if it got stuck
            if output == prev_out and output != 0:
                counter += 1
                if counter >= 3:
                    output = 5

            # Motors actuators
            left_motor = LEVEL[self._level][output]["motors"][0]
            right_motor = LEVEL[self._level][output]["motors"][1]
            time = LEVEL[self._level][output]["time"]

            if prev_out != output or (prev_out == output and output != 0):
                self._robot.move(left_motor, right_motor, time)

            prev_out = output
