from itertools import compress
from random import random

import pandas
import torch
import torch.nn.functional as F

from models.Trainer import SteeringTrainer, BrakingTrainer
from models.basicnetwork import SteeringNet, BrakingNet
from models.data import SteeringTrainingData, BrakingTrainingData
from models.data_extended import ExtendedBrakingData, ExtendedSteeringData
from models.evaluateSteering import SteeringEvaluator, BrakingEvaluator

# steeringNet.load_state_dict(torch.load('models/models/steering/1126145207.model'))

# steeringData_small = pandas.read_csv('training-data/train_data/alpine-1.csv')
# brakingData_small = pandas.read_csv('training-data/train_data/alpine-1.csv')
#
# split_small = [random() < 0.9 for i in range(len(steeringData_small))]
#
# brake_small_train = brakingData_small[split_small]
# brake_small_test = brakingData_small[[not x for x in split_small]]

# steer_small_train = steeringData_small[split_small]
# steer_small_test = steeringData_small[[not x for x in split_small]]

brake_large_train = ExtendedBrakingData(case='train')
brake_large_test = ExtendedBrakingData(case='test')

steer_large_train = ExtendedSteeringData(case='train')
steer_large_test = ExtendedSteeringData(case='test')

# net = SteeringTrainer(extended=False).train(
#     SteeringTrainingData(steer_small_train),
#     torch.optim.Adam,
#     F.mse_loss,
#     10,
#     0.00001,
#     SteeringNet.getPlainNetwork(extended=False)
# )
#
# SteeringEvaluator(SteeringTrainingData(steer_small_test), False).evaluate(net, None, False, 'Steering_small_evaluation')

# net = BrakingTrainer(extended=False).train(
#     BrakingTrainingData(brakingData_small),
#     torch.optim.Adam,
#     F.mse_loss,
#     10,
#     0.00001,
#     net=BrakingNet.getPlainNetwork(False)
# )
#
# BrakingEvaluator(BrakingTrainingData(brake_small_test), False).evaluate(None, net, False, 'Braking_small_evaluation')

#
# net = SteeringTrainer(extended=True).train(
#     steer_large_train,
#     torch.optim.Adam,
#     F.mse_loss,
#     10,
#     0.00001,
#     net=SteeringNet.getPlainNetwork(extended=True)
# )
#
# SteeringEvaluator(steer_large_test, True).evaluate(net, None, True, 'Steering_large_evaluation')


net = BrakingTrainer(extended=True).train(
    brake_large_train,
    torch.optim.Adam,
    F.mse_loss,
    10,
    0.00001
)

BrakingEvaluator(brake_large_test, True).evaluate(None, net, True, 'Braking_large_evaluation')
#
