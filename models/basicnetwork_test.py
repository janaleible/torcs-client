import os

import pandas
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.Trainer import SteeringTrainer
from models.basicnetwork import Net, Meta
from models.data import SteeringTrainingData


trainer = SteeringTrainer()
trainer.train(
    SteeringTrainingData(pandas.read_csv('training-data/train_data/alpine-1.csv')),
    torch.optim.Adam,
    F.mse_loss,
    30,
    0.00001
)

# modelsDir = './models/models/'
#
# dataFile = 'alpine-1'
#
# steeringData = SteeringTrainingData(pandas.read_csv('training-data/train_data/' + dataFile + '.csv'))
#
# optimiser = torch.optim.Adam
# loss = F.mse_loss
# numberOfEpochs = 30
#
# hiddenSize = 60
#
# steeringNet = Net(22, hiddenSize, 1)
# loss = steeringNet.trainNet(steeringData, optimiser, loss, numberOfEpochs, learningRate=0.00001)
#
# modelName = time.strftime('%m%d%H%M%S')
# steeringNet.save(modelsDir, modelName)
#
# # files = [modelsDir + file for file in os.listdir(modelsDir) if file.endswith('.model')]
# # net.load_state_dict(torch.load('models/models/1121123912.model'))
#
# testData = list(steeringData.data.iloc[1])
# testTargets = list(steeringData.targets.iloc[1])
#
# prediction = steeringNet.predict(Variable(torch.FloatTensor(testData)))
#
# modelMeta = {
#     'Expection': str(testTargets),
#     'Prediction': str([value.data.numpy()[0] for value in prediction]),
#     'loss': str(loss),
#     'epochs': str(numberOfEpochs),
#     'optimiser': str(optimiser),
#     'hiddenSize': str(hiddenSize),
#     'data': dataFile
# }
#
# meta = Meta.load()
# meta.append(modelName, modelMeta)
# meta.save()
#
# print('Prediction: ', prediction)
# print('Expected: ', testTargets)
#
# plt.plot(range(len(loss)), loss)
# plt.show()