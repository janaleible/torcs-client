from abc import abstractmethod
from datetime import time
from typing import Callable

import torch
import matplotlib.pyplot as plt

from models.basicnetwork import Net, SteeringNet
from models.data import TrainingData


class Trainer:

    def train(
        self,
        data: TrainingData,
        optimiser: torch.optim,
        loss: Callable,
        numberOfEpochs: int,
        learningRate: float
    ):
        net = self.getNetwork()
        trainingLoss = net.trainNet(data, optimiser, loss, numberOfEpochs, learningRate)
        modelName = 'xx' #time.strftime('%m%d%H%M%S')
        net.save(self.getModelsDir(), modelName)

        plt.plot(range(len(trainingLoss)), trainingLoss)
        plt.show()

    @abstractmethod
    def getNetwork(self) -> Net:
        pass

    @abstractmethod
    def getSubDirectory(self) -> str:
        pass

    def getModelsDir(self):
        return './models/models/'


class SteeringTrainer(Trainer):

    def getNetwork(self) -> Net:
        return SteeringNet.getPlainNetwork()
