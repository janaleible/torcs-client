from abc import abstractmethod

import pandas
import matplotlib.pyplot as plt
from torch.autograd import Variable

from models.Trainer import Trainer
from models.data import SteeringTrainingData, BrakingTrainingData
from models.data_extended import ExtendedSteeringData
from my_driver import MyDriver
from pytocl.car import State, MPS_PER_KMH, DEGREE_PER_RADIANS, Command


class Evaluator():

    def __init__(self, testSet, extended):
        self.extended = extended
        self.testSet = testSet

    @staticmethod
    def dataToState(data) -> State:
        datalist = list(data)
        state = State({
            'angle': datalist[2] / DEGREE_PER_RADIANS,
            'curLapTime': 0,
            'damage': 0,
            'distFromStart': 0,
            'distRaced': 0,
            'fuel': 0,
            'gear': 0,
            'lastLapTime': 0,
            'opponents': {},
            'racePos': 0,
            'rpm': 0,
            'speedX': datalist[0] / MPS_PER_KMH,
            'speedY': 0,
            'speedZ': 0,
            'track': datalist[3:22],
            'trackPos': datalist[1],
            'wheelSpinVel': (0, 0),
            'z': 0,
            'focus': (0,0)
        })
        return state

    @staticmethod
    def dataToStateExtended(data) -> State:
        datalist = list(data)
        state = State({
            'angle': datalist[0],
            'curLapTime': 0,
            'damage': 0,
            'distFromStart': 0,
            'distRaced': 0,
            'fuel': 0,
            'gear': 0,
            'lastLapTime': 0,
            'opponents': {},
            'racePos': 0,
            'rpm': 0,
            'speedX': datalist[1],
            'speedY': datalist[2],
            'speedZ': datalist[3],
            'track': datalist[4:23],
            'trackPos': datalist[23],
            'wheelSpinVel': (
                datalist[24], datalist[25], datalist[26], datalist[27],
            ),
            'z': datalist[28],
            'focus': (
                datalist[29], datalist[30], datalist[31], datalist[32], datalist[33]
            )
        })
        return state

    def evaluate(self, steeringNet, brakingNet, extended, filename):
        observations = []
        driver = MyDriver(steeringNet, brakingNet, extended)

        for i in range(len(self.testSet)):

            state = (Evaluator.dataToStateExtended if self.extended else Evaluator.dataToState)(self.testSet[i][0])

            # sample = Trainer.stateToSample(state, self.extended)

            command = driver.drive(state)
            #
            # observation = {
            #     'target': (brakingData[i][1]).numpy()[0],
            #     'actual': command.brake
            # }

            observation = {
                'target': (self.testSet[i][1]).numpy()[0],
                'actual': self.getPrediction(command)
            }

            observations.append(observation)

        observations.sort(key=lambda observation: observation['target'])

        plt.plot(range(len(observations)), [observation['actual'] for observation in observations], label='predicted')
        plt.plot(range(len(observations)), [observation['target'] for observation in observations], label='target')
        plt.xlabel('Observations')
        plt.ylabel('Prediction')
        plt.legend(loc='upper left')
        plt.savefig(filename + '.pdf')
        plt.close()

    @abstractmethod
    def getPrediction(self, command: Command):
        pass

class BrakingEvaluator(Evaluator):

    def getPrediction(self, command: Command):
        return command.brake

class SteeringEvaluator(Evaluator):

    def getPrediction(self, command: Command):
        return command.steering

# steeringData = SteeringTrainingData(pandas.read_csv('training-data/train_data/alpine-1.csv'))
# brakingData = BrakingTrainingData(pandas.read_csv('training-data/train_data/alpine-1.csv'))

# steeringData = ExtendedSteeringData()

