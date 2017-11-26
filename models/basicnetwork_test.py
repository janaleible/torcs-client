import torch
import torch.nn.functional as F

from models.Trainer import SteeringTrainer, BrakingTrainer
from models.data_extended import ExtendedBrakingData, ExtendedSteeringData

SteeringTrainer(extended=True).train(
    ExtendedSteeringData(),
    torch.optim.Adam,
    F.mse_loss,
    10,
    0.00001
)

# BrakingTrainer(extended=True).train(
#     ExtendedBrakingData(),
#     torch.optim.Adam,
#     F.mse_loss,
#     300,
#     0.00001
# )
