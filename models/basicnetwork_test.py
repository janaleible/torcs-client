import os

import pandas
import torch
from torch.autograd import Variable

from models.basicnetwork import Net, TrainingData

modelsDir = './models/models/'

trainingData = TrainingData(pandas.read_csv('training-data/train_data/aalborg.csv'))
trainingData.append(pandas.read_csv('training-data/train_data/alpine-1.csv'))
trainingData.append(pandas.read_csv('training-data/train_data/f-speedway.csv'))

net = Net(21, 30, 3)
net.trainNet(trainingData, torch.optim.SGD, 2)
net.save(modelsDir)

# files = [modelsDir + file for file in os.listdir(modelsDir) if file.endswith('.model')]
# net.load_state_dict(torch.load(files[-1]))

testData = list(trainingData.data.iloc[1])
testTargets = list(trainingData.targets.iloc[1])

prediction = net(Variable(torch.FloatTensor(testData)))
print('Prediction: ', prediction)
print('Expected: ', testTargets)