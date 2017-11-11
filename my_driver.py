from pytocl.driver import Driver
from pytocl.car import State, Command

import pandas
import torch
from pandas.core.frame import DataFrame
from torch import nn, optim
import torch.nn.functional as Functional
from torch.autograd import Variable

from scratch import CarControllerNet, TrainingData


class MyDriver(Driver):

    def __init__(self):
        super(MyDriver, self).__init__()
        self.net = CarControllerNet()
        self.net.train(
            2,
            optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9),
            TrainingData(pandas.read_csv('training-data/train_data/aalborg.csv'))
        )

    def drive(self, carstate: State) -> Command:

        return self.net.predict(carstate)
