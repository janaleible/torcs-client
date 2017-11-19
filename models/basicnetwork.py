from typing import Callable
import json
import torch
import torch.nn.functional as F
from pandas import DataFrame
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class TrainingData(Dataset):

    targetColumns = ['STEERING']
    dataColumns = ['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS'] + ['TRACK_EDGE_' + str(i) for i in range(19)]

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

    def forward(self, x: Variable) -> Variable:

        x = self.h1(x)
        x = F.sigmoid(x)
        x = self.h2(x)
        x = F.sigmoid(x)

        return x

    def trainNet(self,
        data: TrainingData,
        optimiserFunction: Callable = torch.optim.SGD,
        lossFunction: Callable = F.mse_loss,
        numberOfEpochs: int = 2,
        learningRate: float = 0.001
    ) -> list:

        optimiser = optimiserFunction(self.parameters(), lr=learningRate)

        dataLoader = DataLoader(data, shuffle=True)

        losses = []
        for epoch in range(numberOfEpochs):

            print('Epoch ' + str(epoch))
            totalLoss = 0

            for i, sample in enumerate(dataLoader):
                inputs, targets = sample

                inputs = Variable(inputs)
                targets = Variable(targets)

                optimiser.zero_grad()

                outputs = self(inputs)
                loss = lossFunction(outputs, targets)
                totalLoss += loss
                loss.backward()
                optimiser.step()

            losses.append(totalLoss.data.numpy()[0])

        return losses

    def save(self, directory: str, modelName: str):
        torch.save(self.state_dict(), directory + modelName + '.model')



class Meta():

    def __init__(self):
        self.data = {}
        self.loaded = False

    @staticmethod
    def load(file: str = './models/models/00_meta.json'):
        meta = Meta()
        meta.data = json.load(open(file))
        meta.loaded = True

        return meta

    def append(self, model: str, data: {}):

        if(model in self.data):
            for key, value in data:
                self.data[model][key] = value

        else:
            self.data[model] = data

    def save(self, file: str = './models/models/00_meta.json'):

        if(not self.loaded): raise Exception('Must load meta file first')

        with open(file, 'w') as metaFile:
            json.dump(self.data, metaFile)