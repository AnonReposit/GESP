#!/usr/bin/env python3
"""This is an example to train a task with CMA-ES.
Here it runs CartPole-v1 environment with 100 epoches.
Results:
    AverageReturn: 100
    RiseTime: epoch 38 (itr 760),
              but regression is observed in the course of training.
"""
from garage import wrap_experiment
import garage.sampler.default_worker
from garage.sampler.default_worker import *
from garage._functions import *
import time
import argparse
import sys

TOTAL_COMPUTED_STEPS = 0 # number of episodes (1 episode is equal to a complete begining -> [environment-> action -> reward] -> end cycle)
RUNTIMES = []
START_REF_TIME = None
POPSIZE = 100  # CMA-ES Population size.


parser = argparse.ArgumentParser(description='Run the program')
parser.add_argument('--method', required=True, metavar='method', type=str, help='Must be constant, bestasref, or bestevery.', default=None, nargs='?')
parser.add_argument('--gymEnvName', required=True, metavar='gymEnvName', type=str, help='Gym environment.', default=None, nargs='?')
parser.add_argument('--action_space', required=True, metavar='action_space', type=str, help='Must be continuous or discrete', default=None, nargs='?')
parser.add_argument('--seed', required=True, metavar='seed', type=int, help='Grace time parameter', default=None, nargs='?')
parser.add_argument('--gracetime', required=True, metavar='gracetime', type=int, help='Grace time parameter', default=None, nargs='?')
parser.add_argument('--gens', required=True, metavar='gens', type=int, help='NUmber of generations or epochs', default=None, nargs='?')
parser.add_argument('--max_episode_length', required=True, metavar='max_episode_length', type=int, help='Number of max frames per experiment', default=None, nargs='?')
parser.add_argument('--res_filepath', required=True, metavar='res_filepath', type=str, help='Result file path', default=None, nargs='?')

args = parser.parse_args()


modifyRuntime_method = int(["constant", "bestasref", "bestevery"].index(args.method)) 
gymEnvName = args.gymEnvName

DTU = False
if "DTU" in gymEnvName:
    DTU = True

is_action_space_discrete = bool(["continuous", "discrete"].index(args.action_space))
seed = args.seed
GRACE = args.gracetime
n_epochs = args.gens
MAX_EPISODE_LENGTH = args.max_episode_length
res_filepath = args.res_filepath

REF_CUMULATIVE_FITNESSES = np.array([-1e20] * MAX_EPISODE_LENGTH)
batch_size = 1

print("----")
print(args)
print("----")

EPISODE_INDEX = 0

def rollout(self):
    print("Begin custom rollout")
    
    global REF_CUMULATIVE_FITNESSES
    global GRACE
    global RUNTIMES
    global START_REF_TIME
    global TOTAL_COMPUTED_STEPS
    global EPISODE_INDEX

    if DTU:
        self.env._env.env._terminate_when_unhealthy = False

    if START_REF_TIME is None:
        START_REF_TIME = time.time()


    sum_of_rewards = 0
    episode_start_ref_t = time.time()
    self.start_episode()
    observed_cum_rewards = np.zeros_like(REF_CUMULATIVE_FITNESSES)
    i = -1
    self._max_episode_length = MAX_EPISODE_LENGTH
    was_early_stopped = False
    while not self.step_episode():
        i += 1
        step_reward = self._env_steps[i-1].reward
        sum_of_rewards = sum_of_rewards + step_reward
        observed_cum_rewards[i] = sum_of_rewards
        # Halt computation cumulative reward is worse than ref
        if modifyRuntime_method == 1: # bestasref
            if i >= GRACE and not (   max(observed_cum_rewards[i - GRACE], observed_cum_rewards[i]) >=  min(REF_CUMULATIVE_FITNESSES[i - GRACE], REF_CUMULATIVE_FITNESSES[i])  ):
                print("Stop computation after", i," steps: ref , sum of returns =  ", REF_CUMULATIVE_FITNESSES[i - GRACE], sum_of_rewards)
                was_early_stopped = True
                self._max_episode_length = self._eps_length
        elif modifyRuntime_method == 2: # bestevery
            raise ValueError("This was scraped, because the besrasref method takes into account decreasing objective funcions too!")
                
    self._max_episode_length = MAX_EPISODE_LENGTH       


    print("f =",sum_of_rewards)

    # Updating ref fitness.
    if not was_early_stopped and sum_of_rewards > REF_CUMULATIVE_FITNESSES[-1]:
        print("--Updating refs--")
        print("Old refs:", REF_CUMULATIVE_FITNESSES)
        REF_CUMULATIVE_FITNESSES[0:i] = observed_cum_rewards[0:i]
        REF_CUMULATIVE_FITNESSES[i:] = sum_of_rewards
        print("New refs:", REF_CUMULATIVE_FITNESSES)
        print("--")

    TOTAL_COMPUTED_STEPS += i

    RUNTIMES.append(time.time() - episode_start_ref_t)

    if EPISODE_INDEX%POPSIZE == POPSIZE-1:
        runtimes = "("+";".join(map(str, RUNTIMES))+")"
        with open(res_filepath, "a+") as f:
            print("seed_"+str(seed)+"_gymEnvName_"+gymEnvName, REF_CUMULATIVE_FITNESSES[-1], time.time() - START_REF_TIME, TOTAL_COMPUTED_STEPS, EPISODE_INDEX, runtimes , file=f, sep=",", end="\n")
        RUNTIMES = []


    EPISODE_INDEX += 1
    return self.collect_episode()

# mock rollout function to introduce early stopping
garage.sampler.default_worker.DefaultWorker.rollout = rollout

@wrap_experiment(snapshot_mode="none", log_dir="/tmp/")
def launch_experiment(ctxt=None, gymEnvName=gymEnvName, seed=seed):
    
    import garage.trainer
    # mock save function to avoid wasting time
    garage.trainer.Trainer.save = lambda self, epoch: print("skipp save.")

    from garage.envs import GymEnv
    from garage.experiment.deterministic import set_seed
    from garage.np.algos import CMAES
    from garage.sampler import LocalSampler
    from garage.tf.policies import CategoricalMLPPolicy, ContinuousMLPPolicy
    from garage.trainer import TFTrainer


    set_seed(seed)

    with TFTrainer(ctxt) as trainer:
        global MAX_EPISODE_LENGTH
        if DTU: # DTU  means "Disable terminate_when_unhealthy"
            env = GymEnv(gymEnvName.replace("_DTU", ""), max_episode_length=MAX_EPISODE_LENGTH)
            print("terminate_when_unhealthy will be disabled in each episode. (DTU = True)")
        else:
            env = GymEnv(gymEnvName, max_episode_length=MAX_EPISODE_LENGTH)
            
        
        if is_action_space_discrete:
            policy = CategoricalMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
        else:
            policy = ContinuousMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
        sampler = LocalSampler(agents=policy, envs=env, max_episode_length=env.spec.max_episode_length, is_tf_worker=True)
        algo = CMAES(env_spec=env.spec, policy=policy, sampler=sampler, n_samples=POPSIZE)

        trainer.setup(algo, env)
        trainer.train(n_epochs=n_epochs, batch_size=batch_size)


if __name__ == "__main__":
    launch_experiment(seed=seed)
    
