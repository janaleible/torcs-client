import pandas
import torch
from pandas.core.frame import DataFrame
from torch import nn, optim
import torch.nn.functional as Functional
from torch.autograd import Variable

from pytocl.car import State, Command

targetColumns = ['ACCELERATION', 'BRAKE', 'STEERING']
dataColumns = ['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS'] + ['TRACK_EDGE_' + str(i) for i in range(19)]

numberOfEpochs = 2

aalborg_raw = pandas.read_csv('training-data/train_data/aalborg.csv')
# aalborg_raw = pandas.read_csv('training-data/train_data/test.csv', header = 0, names = targetColumns + dataColumns)

# print(aalborg_raw)

class TrainingData:

    def __init__(self, dataframe: DataFrame):
        self.targets = dataframe.loc[:, targetColumns]
        self.data = dataframe.loc[:, dataColumns]

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, item):
        return torch.FloatTensor(list(self.data.loc[item, :].values)), torch.FloatTensor(list(self.targets.loc[item, :].values))


class Net(nn.Module):

    def __init__(self, inputSize, hiddenSize, outputSize):
        super(Net, self).__init__()

        self.h1 = nn.Linear(inputSize, hiddenSize)
        self.h2 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):

        x = self.h1(x)
        x = Functional.sigmoid(x)
        x = self.h2(x)
        x = Functional.sigmoid(x)

        return x

class CarControllerNet():

    def __init__(self):
        self.trained = False
        self.net = Net(22, 30, 3)

    def train(self, numberOfEpochs, optimiser, trainingData: TrainingData):

        for epoch in range(numberOfEpochs):

            for i in range(len(trainingData)):
                data = trainingData[i]

                inputs, targets = data

                inputs = Variable(inputs)
                targets = Variable(targets)

                optimiser.zero_grad()

                outputs = self.net(inputs)
                loss = Functional.l1_loss(outputs, targets)
                loss.backward()
                optimiser.step()

            self.trained = True

    def predict(self, input: State) -> Command:

        if(not self.trained): raise NotImplementedError

        prediction = self.net(Variable(input))

        command = Command()
        command.accelerator = prediction[0].data.numpy()[0]
        command.brake = prediction[1].data.numpy()[0]
        command.steering = prediction[2].data.numpy()[0]
        command.gear = 1

        return command

    def parameters(self):
        return self.net.parameters()

net = CarControllerNet()
net.train(
    2,
    optim.SGD(net.parameters(), lr=0.001, momentum=0.9),
    TrainingData(pandas.read_csv('training-data/train_data/aalborg.csv'))
)

prediction = net.predict(torch.FloatTensor([-0.0379823, -5.61714E-5, 4.30409E-4, 5.00028, 5.0778, 5.32202, 5.77526, 6.52976, 7.78305, 10.008, 14.6372, 28.8659, 200.0, 28.7221, 14.6009, 9.99199, 7.7742, 6.52431, 5.77175, 5.31976, 5.07646, 4.99972]))

print(prediction)