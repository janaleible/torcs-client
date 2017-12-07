import pickle
from collections import namedtuple

from neat.checkpoint import Checkpointer

population = Checkpointer().restore_checkpoint('neat-checkpoint-218')

Dummy = namedtuple('Dummy', ['fitness'])

winner = Dummy(fitness=0)

for index in population.population.keys():
    individual = population.population[index]
    print(individual.fitness)
    if(individual.fitness and individual.fitness > winner.fitness):
        winner = individual

with open('winner-neat-opponents', 'wb') as file:
    pickle.dump(winner, file)