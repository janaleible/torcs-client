from pytocl.driver import Driver
from pytocl.car import State, Command, DEGREE_PER_RADIANS, MPS_PER_KMH


class MyDriver(Driver):

    timeSinceLastShift = 0

    gearShiftParameters = {
        'shiftDelay': 25,
        'up': 7500,
        'down': 2000
    }

    def __init__(self, net):

        super().__init__()
        self.net = net

    def drive(self, carstate: State) -> Command:

        command = Command()

        command.accelerator = 0.2
        command.gear = self.shiftGears(carstate.gear, carstate.rpm)

        sample = self.state2sample(carstate)

        command.steering = (self.net.activate(sample)[0]) - 0.5

        return command

    def state2sample(self, carstate: State):

        sample = []
        sample.append(carstate.angle / DEGREE_PER_RADIANS)
        sample.append(carstate.distance_from_center)
        sample.append(carstate.speed_x / MPS_PER_KMH)
        [sample.append(distance) for distance in carstate.distances_from_edge]
        # [sample.append(sum(carstate.opponents[6*i:6*i+5])) for i in range(6)]
        # sample.append(carstate.z)
        return sample

    def shiftGears(self, previousGear: int, rpm: float) -> int:

        if (previousGear <= 0):
            newGear = 1
        elif (self.timeSinceLastShift < self.gearShiftParameters['shiftDelay']):
            newGear = previousGear
        else:
            if (rpm > self.gearShiftParameters['up']):
                newGear = min(previousGear + 1, 6)
            elif (rpm < self.gearShiftParameters['down']):
                newGear = max(previousGear - 1, 1)
            else:
                newGear = previousGear

        if (previousGear != newGear):
            self.timeSinceLastShift = 0
        else:
            self.timeSinceLastShift += 1

        return newGear
