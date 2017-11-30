#! /usr/bin/env python3
import neat
import subprocess
import pickle
from neat.checkpoint import Checkpointer

from FileReporter import FileReporter
from pytocl.main import main
from my_driver import MyDriver

# if __name__ == '__main__':
#     main(MyDriver(None))

def eval_genome(genome, config):

    net = neat.ctrnn.CTRNN.create(genome, config, 10)

    subprocess.call('neat/autostart.sh', shell=True)

    main(MyDriver(net))

    with open('neat/fitnessFile', 'r') as fitnessFile:
        fitness = float(fitnessFile.read())

    subprocess.call('neat/autostop.sh', shell=True)
    print('fitness: {}'.format(fitness))

    return fitness

def eval_genomes(genomes, config):

    for genomeID, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run():

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'neat/config-ctrnn')

    population = Checkpointer().restore_checkpoint('neat-checkpoint-15')
#    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(FileReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(Checkpointer(generation_interval=1))

    winner = population.run(eval_genomes, 3000)

    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open('winner-neat', 'wb') as file:
        pickle.dump(winner, file)

    print(winner)

if __name__ == '__main__':
    run()