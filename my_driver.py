from pytocl.driver import Driver
from pytocl.car import State, Command


class MyDriver(Driver):



    def drive(self, carstate: State) -> Command:

        command = Command()
        return command
