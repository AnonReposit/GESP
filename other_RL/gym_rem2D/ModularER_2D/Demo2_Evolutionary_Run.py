import gym_rem2D.envs.Modular2DEnv
import REM2D_main as r2d
import datetime
import os
import argparse
import numpy as np
import time

# from other_RL.meta_world_and_garage.test_example_garage_cart_pole_CMA_ES import MAX_EPISODE_LENGTH

# number of steps. One episode contains many steps: begining -> ... step repeat ([environment-> action -> reward]) ... -> end cycle),
global_vars = {
"TOTAL_COMPUTED_STEPS":0,
"RUNTIMES":[],
"START_REF_TIME":None,
"EPISODE_INDEX":0,
"POPSIZE":100, 
"MAX_EPISODE_LENGTH":10000,
"EPISODE_STASRT_REF_T":None,
"TOTAL_COMPUTED_STEPS":0,
"STEPS_CURRENT":0,
}

class stopwatch:

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_t = time.time()
        self.pause_t=0

    def pause(self):
        self.pause_start = time.time()
        self.paused=True

    def resume(self):
        if self.paused:
            self.pause_t += time.time() - self.pause_start
            self.paused = False

    def get_time(self):
        return time.time() - self.start_t - self.pause_t


STOPWATCH = stopwatch()
r2d.STOPWATCH = STOPWATCH

parser = argparse.ArgumentParser(description='Run the program')
parser.add_argument('--method', required=True, metavar='method', type=str, help='Must be constant, bestasref, or bestevery.', default=None, nargs='?')
parser.add_argument('--seed', required=True, metavar='seed', type=int, help='Grace time parameter', default=None, nargs='?')
parser.add_argument('--gracetime', required=True, metavar='gracetime', type=int, help='Grace time parameter', default=None, nargs='?')
parser.add_argument('--res_filepath', required=True, metavar='res_filepath', type=str, help='Result file path', default=None, nargs='?')

args = parser.parse_args()


method = args.method 
seed = args.seed
GRACE = args.gracetime
res_filepath = args.res_filepath

global_vars["REF_FITNESSES"] = np.array([-1e20] * global_vars["MAX_EPISODE_LENGTH"])
global_vars["OBSERVED_FITNESSES"] = np.array([-1e20] * global_vars["MAX_EPISODE_LENGTH"])

print("----")
print(args)
print("----")




def wod_update_nokill(self):
	pass







steps_current = 0
def callback_en_of_step(self, done, fitness, global_vars):


	i = self.current_steps
	global_vars["OBSERVED_FITNESSES"][i] = fitness
	# Halt computation cumulative reward is worse than ref
	if method == "bestasref":
		if i >= GRACE and global_vars["REF_FITNESSES"][i - GRACE] > fitness:
			# print("Stop computation after", i," steps: ref , observed =  ", REF_FITNESSES[i - GRACE], fitness)
			self.wod.position = 10000000.0 # increase wall of death substantially


	# print(done, self.current_steps, self.total_steps)
	# print(fitness)

	global_vars["TOTAL_COMPUTED_STEPS"] += 1
	global_vars["STEPS_CURRENT"] += 1
	if done:

		print(global_vars)

		print("Done in callback step, with total",global_vars["TOTAL_COMPUTED_STEPS"],"steps.")
		print(f"Used ",global_vars["STEPS_CURRENT"]," steps.")
		global_vars["STEPS_CURRENT"] = 0
		# Updating ref fitness.
		if fitness > global_vars["REF_FITNESSES"][-1]:
			print("--Updating refs--")
			print("Old refs:", global_vars["REF_FITNESSES"])
			global_vars["REF_FITNESSES"][0:i] = global_vars["OBSERVED_FITNESSES"][0:i]
			global_vars["REF_FITNESSES"][i:] = fitness
			print("New refs:", global_vars["REF_FITNESSES"])
			print("--")

		global_vars["RUNTIMES"].append(0)
		return global_vars






def callback_end_of_gen(self, global_vars):

	print(global_vars)

	best_fitness_test = self.ev_best(global_vars)


	runtimes = "("+";".join(map(str, global_vars["RUNTIMES"]))+")"
	print("Saving results in ", res_filepath)
	with open(res_filepath, "a+") as f:
		print("seed_"+str(seed), global_vars["REF_FITNESSES"][-1], STOPWATCH.get_time(), global_vars["TOTAL_COMPUTED_STEPS"], global_vars["EPISODE_INDEX"], runtimes, best_fitness_test, file=f, sep=",", end="\n")
	global_vars["RUNTIMES"] = []
	global_vars["EPISODE_INDEX"] += 1



def run_experiment(exp_dir, global_vars):
	config, dir = r2d.setup(directory=exp_dir)	
	experiment = r2d.run2D(config,dir)
	experiment.run(config, global_vars)


	print("Evaluating best: ")
	best_f_test = experiment.ev_best(global_vars)
	with open(res_filepath, "a+") as f:
		print(best_f_test, file=f, end="\n")
	
if __name__=="__main__":
	unique_number = str(datetime.datetime.now()).replace("-","").replace(".","").replace(" ","").replace(":","")
	if method=="problemspecific":
		pass
	elif method=="bestasref":
		gym_rem2D.envs.Modular2DEnv.WallOfDeath.update = wod_update_nokill
	elif method=="nokill":
		gym_rem2D.envs.Modular2DEnv.WallOfDeath.update = wod_update_nokill

	gym_rem2D.envs.Modular2DEnv.Modular2D.callback_en_of_step = callback_en_of_step
	r2d.run2D.callback_end_of_gen = callback_end_of_gen

	
	exp_dir = "/tmp/test" + unique_number  + "/"
	os.makedirs(exp_dir)
	run_experiment(exp_dir, global_vars)
