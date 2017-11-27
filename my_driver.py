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

    stats = pandas.read_csv('./training-data/train_data/allstats.csv')

    def drive(self, carstate: State) -> Command:

        command = Command()

        steeringNet = SteeringNet.getPlainNetwork(extended=True)
        steeringNet.load_state_dict(torch.load('models/models/steering/1126145207.model'))

        # brakingNet = BrakingNet.getPlainNetwork()
        # brakingNet.load_state_dict(torch.load('models/models/braking/1124142419.model'))

        steeringSample = Trainer.stateToSample(carstate, extended=True)
        brakingSample = Trainer.stateToSample(carstate, extended=True)

        command.steering = steeringNet.predict(steeringSample)
        # command.brake = brakingNet.predict(brakingSample)
        command.accelerator = 0.2

        command.gear = self.shiftGears(carstate.gear, carstate.rpm)

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
