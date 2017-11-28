#! /usr/bin/env python3
import neat
import subprocess
import pickle
from neat.checkpoint import Checkpointer

from pytocl.main import main
from my_driver import MyDriver

# if __name__ == '__main__':
#     main(MyDriver(None))

def eval_genome(genome, config):

    net = neat.ctrnn.CTRNN.create(genome, config, 10)

    subprocess.call('./autostart.sh', shell=True)

    main(MyDriver(net))

    with open('neat/fitnessFile', 'r') as fitnessFile:
        fitness = float(fitnessFile.read())

    subprocess.call('./autostop.sh', shell=True)
    print('fitness: {}'.format(fitness))

    return fitness

def eval_genomes(genomes, config):

    for genomeID, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run():

    stats = neat.StatisticsReporter()
    check = Checkpointer(2, time_interval_seconds=None)

    try:
        checkpoint = Checkpointer().restore_checkpoint('neat/neat-checkpoint-1')
    except:
        checkpoint = Checkpointer()

    checkpoint.add_reporter(stats)
    checkpoint.add_reporter(check)
    checkpoint.add_reporter(neat.StdOutReporter(True))

    evaluator = neat.ParallelEvaluator(1, eval_genomes)
    winner = checkpoint.run(evaluator.evaluate(100))

    with open('winner-neat', 'wb') as file:
        pickle.dump(winner, file)

    print(winner)

if __name__ == '__main__':
    run()