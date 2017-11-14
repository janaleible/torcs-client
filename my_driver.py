from pytocl.driver import Driver
from pytocl.car import State, Command


class MyDriver(Driver):
    # Override the `drive` method to create your own driver

    timeSinceLastShift = 0

    gearShiftParameters = {
        'shiftDelay': 25,
        'up': 7500,
        'down': 2000
    }

    def drive(self, carstate: State) -> Command:

        command = Command()

        command.accelerator = 1
        command.gear = self.shiftGears(carstate.gear, carstate.rpm)

        return command

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