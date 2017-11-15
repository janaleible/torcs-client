import torch
from torch.autograd import Variable

from pytocl.driver import Driver
from pytocl.car import State, Command

from models.basicnetwork import Net


class MyDriver(Driver):

    timeSinceLastShift = 0

    gearShiftParameters = {
        'shiftDelay': 25,
        'up': 7500,
        'down': 2000
    }

    def drive(self, carstate: State) -> Command:

        command = Command()

        net = Net(21, 30, 3)
        net.load_state_dict(torch.load('models/models/11151134.model'))

        prediction = net(self.stateToSample(carstate))

        command.accelerator = self.predictionToFloat(prediction[0])
        command.brake = self.predictionToFloat(prediction[1])
        command.steering = self.predictionToFloat(prediction[2])

        command.gear = self.shiftGears(carstate.gear, carstate.rpm)

        return command

    def predictionToFloat(self, prediction: Variable) -> float:
        return prediction.data.numpy()[0]

    def stateToSample(self, state: State) -> Variable:

        sample = [state.speed_x, state.distance_from_center, state.angle] + [distance for distance in state.distances_from_edge]

        return Variable(torch.FloatTensor(sample[0:-1]))

    def shiftGears(self, previousGear: int, rpm: float) -> int:

        if (previousGear == 0): newGear = 1
        elif(self.timeSinceLastShift < self.gearShiftParameters['shiftDelay']): newGear = previousGear
        else:
            if (rpm > self.gearShiftParameters['up']): newGear = max(previousGear + 1, 6)
            elif (rpm < self.gearShiftParameters['down']): newGear = previousGear - 1
            else: newGear = previousGear

        if(previousGear != newGear): self.timeSinceLastShift = 0
        else: self.timeSinceLastShift += 1

        return newGear
