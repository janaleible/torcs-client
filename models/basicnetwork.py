from typing import Callable
import time
import torch
import torch.nn.functional as F
from pandas import DataFrame
from torch.autograd import Variable
from torch.utils.data import Dataset


class TrainingData(Dataset):

    targetColumns = ['ACCELERATION', 'BRAKE', 'STEERING']
    dataColumns = ['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS'] + ['TRACK_EDGE_' + str(i) for i in range(18)]

    def __init__(self, dataframe: DataFrame = None):
        self.targets = DataFrame()
        self.data = DataFrame()

        if(dataframe is not None): self.append(dataframe)

    def append(self, dataframe: DataFrame):
        dataframe.index = range(self.__len__(), self.__len__() + len(dataframe))
        self.targets = self.targets.append(dataframe.loc[:, TrainingData.targetColumns])
        self.data = self.data.append(dataframe.loc[:, TrainingData.dataColumns])

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, item: int) -> (torch.FloatTensor, torch.FloatTensor):
        return torch.FloatTensor(list(self.data.loc[item, :].values)), torch.FloatTensor(list(self.targets.loc[item, :].values))


class Net(torch.nn.Module):

    def __init__(self, inputSize: int, hiddenSize: int, outputSize: int):
        super(Net, self).__init__()

        self.h1 = torch.nn.Linear(inputSize, hiddenSize)
        self.h2 = torch.nn.Linear(hiddenSize, outputSize)

    def forward(self, x):

        x = self.h1(x)
        x = F.sigmoid(x)
        x = self.h2(x)
        x = F.sigmoid(x)

        return x

    def trainNet(self, data: TrainingData, optimiserFunction: Callable, numberOfEpochs: int):
        
        optimiser = optimiserFunction(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(numberOfEpochs):
            for i in range(len(data)):
                inputs, targets = data[i]

                inputs = Variable(inputs)
                targets = Variable(targets)

                optimiser.zero_grad()

                outputs = self(inputs)
                loss = F.l1_loss(outputs, targets)
                loss.backward()
                optimiser.step()

    def save(self, directory: str):
        torch.save(self.state_dict(), directory + time.strftime('%m%d%H%M%S') + '.model')
