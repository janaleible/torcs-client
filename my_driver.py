import pandas
import torch

from models.Trainer import Trainer
from pytocl.driver import Driver
from pytocl.car import State, Command

from models.basicnetwork import SteeringNet, BrakingNet


class MyDriver(Driver):

    timeSinceLastShift = 0

    gearShiftParameters = {
        'shiftDelay': 25,
        'up': 7500,
        'down': 2000
    }

    def __init__(self, steeringNet, brakingNet, extended):
        self.steeringNet = steeringNet
        self.brakingNet = brakingNet
        self.extended = extended

        if steeringNet is None:
            steeringNet = SteeringNet.getPlainNetwork(extended=True)
            steeringNet.load_state_dict(torch.load('models/models/steering/1206112224.model'))
            self.steeringNet = steeringNet



    def drive(self, carstate: State) -> Command:

        command = Command()


        # brakingNet = BrakingNet.getPlainNetwork()
        # brakingNet.load_state_dict(torch.load('models/models/braking/1124142419.model'))

        steeringSample = Trainer.stateToSample(carstate, True)
        # brakingSample = Trainer.stateToSample(carstate, self.extended)


        command.steering = self.steeringNet.predict(steeringSample)
        # if self.brakingNet is not None: command.brake = self.brakingNet.predict(brakingSample)
        command.accelerator = 0.2

        command.gear = self.shiftGears(carstate.gear, carstate.rpm)
        print(command)

        return command

    def recoveryCommand(self) -> Command:

        command = Command()

        command.gear = -1
        command.accelerator = 0.1
        command.steering = 0

        return command

    def shiftGears(self, previousGear: int, rpm: float) -> int:

        if (previousGear <= 0): newGear = 1
        elif(self.timeSinceLastShift < self.gearShiftParameters['shiftDelay']): newGear = previousGear
        else:
            if (rpm > self.gearShiftParameters['up']): newGear = min(previousGear + 1, 6)
            elif (rpm < self.gearShiftParameters['down']): newGear = max(previousGear - 1, 1)
            else: newGear = previousGear

        if(previousGear != newGear): self.timeSinceLastShift = 0
        else: self.timeSinceLastShift += 1

        return newGear

    def my_accelerate(self, speed) -> float:

        if speed < 20: return 1
        else: return min(0.2, -0.02 * speed + 1.8)
