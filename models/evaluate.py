import math
import pandas
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable

from models.basicnetwork import Net, TrainingData
from my_driver import MyDriver


def stateToSample(state) -> Variable:
    sample = [
        (state[0] - stats['SPEED'][0]) / stats['SPEED'][1],
        (state[1] - stats['TRACK_POSITION'][0]) / stats['TRACK_POSITION'][1],
        (radiansToDegrees(state[2]) - stats['ANGLE_TO_TRACK_AXIS'][0]) /
        stats['ANGLE_TO_TRACK_AXIS'][1]
    ] + [
        (distance - stats['TRACK_EDGE_' + str(i)][0]) / stats['TRACK_EDGE_' + str(i)][1]
        for i, distance in enumerate(state[3:22])
    ]

    return Variable(torch.FloatTensor(sample))

def degToRadians(degree: float) -> float:
    return (degree * math.pi) / 180

def radiansToDegrees(radians: float) -> float:
    return (radians * 180) / math.pi

def predictionToFloat(prediction: Variable) -> float:
    return prediction.data.numpy()[0]

net = Net(22, 60, 2)
net.load_state_dict(torch.load('models/models/1118113406.model'))

stats = pandas.read_csv('./training-data/train_data/allstats.csv')

trainingData = TrainingData(pandas.read_csv('training-data/train_data/allnormalized.csv'))

steering = trainingData.targets['STEERING']

observations = []

for i in range(len(trainingData)):

    sample = stateToSample(trainingData[i][0])

    observation = {
        'target': (trainingData[i][1])[1],
        'actual': predictionToFloat(net(sample)[1])
    }

    observations.append(observation)

observations.sort(key=lambda observation: observation['target'])

plt.plot(range(len(observations)), [observation['target'] for observation in observations])
plt.plot(range(len(observations)), [observation['actual'] for observation in observations])
plt.show()
