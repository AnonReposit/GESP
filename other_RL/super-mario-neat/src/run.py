import neat
import pickle
import sys
sys.path.append("other_RL/gym-super-mario-master")
import gym, ppaquette_gym_super_mario
import visualize
import gzip
import neat.genome
from joblib import Parallel, delayed
import time

ACTIONS = [
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 1],
]
LEVEL = "6-2"
FILENAME_nokill = "./results/data/super_mario/level_6-2_nokill_2.pkl"
FILENAME_constant = "./results/data/super_mario/level_6-2_constant_2.pkl"
FILENAME_bestasref = "./results/data/super_mario/level_6-2_bestasref_2.pkl"
CONFIG = './other_RL/super-mario-neat/src/config'



def main(config_file, file, level=LEVEL):
    # with gzip.open(FILENAME) as f:
    #   config = pickle.load(f)[1]
    # print(str(config.genome_type.size))
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    genome = pickle.load(open(file, 'rb'))
    env = gym.make('ppaquette/SuperMarioBros-'+level+'-Tiles-v0')
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    info = {'distance': 0}
    try:
        state = env.reset()
        done = False
        i = 0
        old = 40
        while not done:
            state = state.reshape(208)
            output = net.activate(state)
            ind = output.index(max(output))
            s, reward, done, info = env.step(ACTIONS[ind])
            state = s
            i += 1
            if i % 50 == 0:
                if old == info['distance']:
                    break

                else:
                    old = info['distance']
        print("Distance: {}".format(info['distance']))
        env.close()
        return info['distance'], i
    except KeyboardInterrupt:
        env.close()
        exit()


if __name__ == "__main__":

        ref = time.time()
        main(CONFIG, FILENAME_nokill)
        print("Took ", time.time() - ref, "seconds")
        ref = time.time()
        main(CONFIG, FILENAME_constant)
        print("Took ", time.time() - ref, "seconds")
        ref = time.time()
        main(CONFIG, FILENAME_bestasref)
        print("Took ", time.time() - ref, "seconds")
        ref = time.time()
