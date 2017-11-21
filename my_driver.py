import math

import pandas
import torch
from torch.autograd import Variable

from pytocl.driver import Driver
from pytocl.car import State, Command

from models.basicnetwork import Net, SteeringNet


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

        steeringNet = SteeringNet(22, 60, 1)
        steeringNet.load_state_dict(torch.load('models/models/1121123912.model'))

        # command.accelerator = self.predictionToFloat(prediction[0])
        # command.brake = self.predictionToFloat(prediction[0])
        command.steering = steeringNet.predict(self.stateToSample(carstate))
        command.accelerator = 0.2

        # print(command)

        command.gear = self.shiftGears(carstate.gear, carstate.rpm)
        # command.gear = 1

        return command


    def stateToSampleNormalised(self, state: State) -> Variable:

        sample = [
             (state.speed_x - self.stats['SPEED'][0]) / self.stats['SPEED'][1],
             (state.distance_from_center - self.stats['TRACK_POSITION'][0]) / self.stats['TRACK_POSITION'][1],
             (state.angle - self.stats['ANGLE_TO_TRACK_AXIS'][0]) / self.stats['ANGLE_TO_TRACK_AXIS'][1]
        ] + [
            (distance - self.stats['TRACK_EDGE_' + str(i)][0]) / self.stats['TRACK_EDGE_' + str(i)][1]
            for i, distance in enumerate(state.distances_from_edge)
        ]

        return Variable(torch.FloatTensor(sample))

    def stateToSample(self, state: State) -> Variable:

        sample = [
             state.speed_x,
             state.distance_from_center,
             self.degToRadians(state.angle)
        ] + [
            distance
            for i, distance in enumerate(state.distances_from_edge)
        ]

        return Variable(torch.FloatTensor(sample))

    def shiftGears(self, previousGear: int, rpm: float) -> int:

        if (previousGear == 0): newGear = 1
        elif(self.timeSinceLastShift < self.gearShiftParameters['shiftDelay']): newGear = previousGear
        else:
            if (rpm > self.gearShiftParameters['up']): newGear = min(previousGear + 1, 6)
            elif (rpm < self.gearShiftParameters['down']): newGear = max(previousGear - 1, 1)
            else: newGear = previousGear

        if(previousGear != newGear): self.timeSinceLastShift = 0
        else: self.timeSinceLastShift += 1

        return newGear

    def degToRadians(self, degree: float) -> float:
        return (degree * math.pi) / 180

    def radiansToDegrees(self, radians: float) -> float:
        return (radians * 180) / math.pi