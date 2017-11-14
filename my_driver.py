from pytocl.driver import Driver
from pytocl.car import State, Command


class MyDriver(Driver):
    # Override the `drive` method to create your own driver

    def drive(self, carstate: State) -> Command:

        command = Command()

        command.accelerator = 1
        command.gear = 1

        return command
