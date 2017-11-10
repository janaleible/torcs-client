import pandas
import torch
from pandas.core.frame import DataFrame
from torch import nn, optim
import torch.nn.functional as Functional
from torch.autograd import Variable

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

net = Net(22, 30, 3)

optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(numberOfEpochs):

    runningLoss = 0.0

    train = TrainingData(aalborg_raw)
    for i in range(len(train)):

        data = train[i]

        inputs, targets = data

        inputs = Variable(inputs)
        targets = Variable(targets)

        optimiser.zero_grad()

        outputs = net(inputs)
        loss = Functional.l1_loss(outputs, targets)
        loss.backward()
        optimiser.step()

        runningLoss += loss.data[0]

        if(i % 2000 == 1999):
            print('[%d, %5d] loss: %.3f' % (
                epoch + 1,
                i + 1,
                runningLoss / 2000
            ))

print('Done training')