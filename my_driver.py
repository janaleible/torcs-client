import pickle
from collections import defaultdict

import math

import neat

from pytocl.driver import Driver
from pytocl.car import State, Command, DEGREE_PER_RADIANS, MPS_PER_KMH


class MyDriver(Driver):

    timeSinceLastShift = 0

    gearShiftParameters = {
        'shiftDelay': 25,
        'up': 7500,
        'down': 2000
    }

    roadmap = []

    def __init__(self, net):

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             'neat/config-ctrnn')

        super().__init__()
        if not net is None: self.net = net
        else:
            with open('winner-neat-opponents', 'rb') as file:
                # unpickler = pickle.Unpickler(file)
                # pickled = unpickler.load()
                pickled = pickle.load(file)
                self.net = neat.nn.FeedForwardNetwork.create(pickled, config)
                
        self.state = 'normal'

        # with open('roadmap', 'wb') as file:
        #     pickle.dump([], file)

    def drive(self, carstate: State) -> Command:

        command = Command()


        if self.state == 'reverse':
            command.steering = 1
            command.accelerator = 0.2
            command.gear = 1

        elif self.state == 'off-track-left':
            command.steering = -1
            command.gear = -1
            command.accelerator = 0.5

        elif self.state == 'off-track-right':
            command.steering = 1
            command.gear = -1
            command.accelerator = 0.5

        else:
            sample = self.state2sample(carstate)

            result = self.net.activate(sample)

            command.accelerator = self.my_accelerate(carstate.speed_x)
            command.gear = self.shiftGears(carstate.gear, carstate.rpm)
            command.steering = result[0] - 0.5
            command.brake = 0

            interval = 10
            position = math.floor(int(carstate.distance_from_start) / interval)

            # with open('roadmap', 'rb') as file:
            #     self.roadmap = pickle.load(file)
            #
            # if position + 1 < len(self.roadmap):
            #     if max(self.roadmap[position:position + int(200 / interval)]) < 0.1:
            #         command.accelerator = 1
            #         print('speed up')
            #     else:
            #         if carstate.speed_x > 80:
            #             command.brake = 1
            #             command.accelerator = 0
            #             print('slow down')
            #
            #
            # if len(self.roadmap) == position and int(carstate.distance_from_start) % interval == 0:
            #     self.roadmap.append(abs(command.steering))
            #     print('Pos: {}, len(RM): {}'.format(position, len(self.roadmap)))
            #     # print(self.roadmap)
            #     with open('roadmap', 'wb') as file:
            #         pickle.dump(self.roadmap, file)
            #
            # print('brake: {}, acc: {}, steering: {}'.format(command.brake, command.accelerator, command.steering))


        # if self.state == 'off-track-left' and all(distance > 0 for distance in carstate.distances_from_edge): self.state = 'normal'
        # elif self.state == 'off-track-right' and all(distance > 0 for distance in carstate.distances_from_edge): self.state = 'normal'
        # elif carstate.angle > 90 and carstate.angle < 270: self.state = 'reverse'
        # elif all(distance < 0 for distance in carstate.distances_from_edge):
        #     if carstate.angle < 180: self.state = 'off-track-right'
        #     else: self.state = 'off-track-left'
        #
        #
        # else: self.state = 'normal'

        return command

    def state2sample(self, carstate: State):

        sample = []
        sample.append(carstate.angle / DEGREE_PER_RADIANS)
        sample.append(carstate.distance_from_center)
        #sample.append(carstate.speed_x / MPS_PER_KMH)
        # sample.append(80)
        [sample.append(distance) for distance in carstate.distances_from_edge]
        [sample.append(sum(carstate.opponents[6*i:6*i+5])) for i in range(6)]
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

    def my_accelerate(self, speed) -> float:

        if speed < 20: return 1
        else: return min(0.2, -0.02 * speed + 1.8)

