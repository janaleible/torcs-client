import os

import pandas
import torch
from torch.autograd import Variable

from models.basicnetwork import Net, TrainingData

modelsDir = './models/models/'

aalborg = TrainingData(pandas.read_csv('training-data/train_data/aalborg.csv'))
# aalborg_raw = pandas.read_csv('training-data/train_data/test.csv', header = 0, names = targetColumns + dataColumns)

net = Net(21, 30, 3)
# net.trainNet(aalborg, torch.optim.SGD, 25)
# net.save(modelsDir)

files = [modelsDir + file for file in os.listdir(modelsDir) if file.endswith('.model')]
net.load_state_dict(torch.load(files[-1]))

testData = list(aalborg.data.iloc[1])
testTargets = list(aalborg.targets.iloc[1])

prediction = net(Variable(torch.FloatTensor(testData)))
print('Prediction: ', prediction)
print('Expected: ', testTargets)