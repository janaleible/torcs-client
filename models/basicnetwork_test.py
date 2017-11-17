import os

import pandas
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from models.basicnetwork import Net, TrainingData, Meta

modelsDir = './models/models/'

trainingData = TrainingData(pandas.read_csv('training-data/train_data/aalborgnormalized.csv'))
trainingData.append(pandas.read_csv('training-data/train_data/alpine-1normalized.csv'))
trainingData.append(pandas.read_csv('training-data/train_data/f-speedwaynormalized.csv'))

optimiser = torch.optim.SGD
loss = F.mse_loss
numberOfEpochs = 20

net = Net(21, 30, 3)
net.trainNet(trainingData, optimiser, loss, numberOfEpochs, weight_decay=0.1)

modelName = time.strftime('%m%d%H%M%S')
net.save(modelsDir, modelName)



# files = [modelsDir + file for file in os.listdir(modelsDir) if file.endswith('.model')]
# net.load_state_dict(torch.load(files[-1]))

testData = list(trainingData.data.iloc[1])
testTargets = list(trainingData.targets.iloc[1])

prediction = net(Variable(torch.FloatTensor(testData)))


modelMeta = {
    'Expection': str(testTargets),
    'Prediction': str([value.data.numpy()[0] for value in prediction]),
    'loss': str(loss),
    'epochs': str(numberOfEpochs),
    'optimiser': str(optimiser)
}

meta = Meta.load()
meta.append(modelName, modelMeta)
meta.save()

print('Prediction: ', prediction)
print('Expected: ', testTargets)