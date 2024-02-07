from argparse import ArgumentError
import pandas as pd
from utils.UpdateParameter import *
import subprocess
import time
import re
from os.path import exists
import sys
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed
from tqdm import tqdm as tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Dirty hack to import ./other_RL/super-mario-neat/src/run.py
import pathlib
sys.path.append("./other_RL/super-mario-neat/src") 
import run

gens = 100
seeds = list(range(2,32))
gracetime = 40
fincrementsize = 150
parallel_threads = 8

savefig_paths = ["results/figures", "../paper/images/supermario/"]

methods = ["nokill", "bestasref", "constant"]
method_plot_name_list = ["Standard", "GESP", "Problem Specific"]
task_list = ["1-4", "2-1", "4-1", "4-2", "5-1", "6-2", "6-4"]
apply_legend_to_first_plot = True 

if sys.argv[1] == "--plot":

    import numpy as np
    import pandas as pd
    pd.options.mode.chained_assignment = None 
    class plot_time_evals:

        quantiles = np.linspace(0,1,20)
        
        def __init__(self):
            self.df = pd.DataFrame(columns=["task", "method", "seed", "time", "evals", "evals_per_second"])
            pass

        def add_data(self, task, method, seed, time, evals):
            self.df = self.df.append({"task":task, "method":method,"seed":seed,"time":time,"evals":evals, "evals_per_second":float(evals) / time}, ignore_index=True)

        # Get how many more solutions method2 evaluates than method1 in the same amount of simulation time.
        def get_proportion(self, task, method1, method2):

            # # Start interactive mode for debug debugging
            # import code; code.interact(local=locals())

            time_low = max(
                    np.quantile(self.df.query(f"task == '{task}' and method == '{method1}'").time, 0.05),
                    np.quantile(self.df.query(f"task == '{task}' and method == '{method2}'").time, 0.05)
                    )
            time_up = min(
                    np.quantile(self.df.query(f"task == '{task}' and method == '{method1}'").time, 0.95),
                    np.quantile(self.df.query(f"task == '{task}' and method == '{method2}'").time, 0.95)
                    )
            
            x_time = self.quantiles * (time_up - time_low) + time_low

            proportions = np.zeros_like(x_time)
            for i in range(len(x_time)):
                rows = self.df.iloc[self.df.query(f"time > {x_time[i]} and task == '{task}'").groupby(["seed","method"])['time'].idxmin()]
                rows = rows.groupby(["method"]).mean()
                proportions[i] = rows.query(f"method == '{method2}'").evals_per_second[0] / rows.query(f"method == '{method1}'").evals_per_second[0]

            return self.quantiles, proportions

    pe = plot_time_evals()




index = -1
for task in tqdm(task_list):
    index += 1

    if len(sys.argv) != 2:
        raise ArgumentError("this script requires only one argument --plot --launch_local")

    if sys.argv[1] not in ("--plot", "--launch_local", "--tgrace_nokill", "--tgrace_different_values"):
        raise ArgumentError("this script requires only one argument --plot --launch_local --tgrace_nokill --tgrace_different_values")


    #region local_launch

    if sys.argv[1] == "--launch_local":
        import itertools
        import time
        from os.path import exists
        experiment_parameters = list(itertools.product(seeds, methods, task_list))

        def run_with_experiment_index(experiment_index):

            seed, method, task = experiment_parameters[experiment_index]
            print(seed, method, task)
            time.sleep(0.5)
            print(f"Launching with seed {seed} in experiment_halveruntime.py ...")

            # # Reduce evaluations not needed in nokill if we kill all experiments in 1k iterations.
            #new_gens = str(gens if method != "nokill" else gens // 6)
            resultfilename = f"results/data/super_mario/supermario{task}_{seed}.txt"
            cmd_str = f"python3 other_RL/super-mario-neat/src/main.py train --gen 10000 --task {task} --seed {seed} --method {method} --resultfilename {resultfilename} --gracetime {gracetime}"
            import os
            try:
                os.remove(resultfilename)
                print(f"removed old log {resultfilename}")
            except FileExistsError:
                pass
            except FileNotFoundError:
                pass
            cmd_str = f"python3 other_RL/super-mario-neat/src/main.py train --gen 10000 --task {task} --seed {seed} --method {method} --resultfilename {resultfilename} --gracetime {gracetime}"
            exec_res=subprocess.run(cmd_str,shell=True, capture_output=True)
        
        Parallel(n_jobs=parallel_threads, verbose=12)(delayed(run_with_experiment_index)(i) for i in range(len(experiment_parameters)))
        print("Finished trainig controllers. Now we measure the runtime of the best solutions in each case.")
        for method in methods:
            for seed in seeds:
                ref = time.time()
                pickle_network_path = "./results/data/super_mario/task_{task}_{method}_{seed}.pkl"
                if exists(f"./results/data/super_mario/task_{task}_{method}_{seed}.pkl"):
                    res, frames = run.main(run.CONFIG, f"./results/data/super_mario/task_{task}_{method}_{seed}.pkl")
                    print(task, method, seed, res, frames)
                    with open("results/data/super_mario/runtimes.csv", "a+") as f:
                        print(task, method, seed, res, frames, sep=",", file=f)
        exit(0)




    #region local_launch
    if sys.argv[1] == "--tgrace_different_values":
        import itertools
        import time

        seeds = list(range(2,32))
        max_optimization_time = 1400.0

        method = "tgraceexpdifferentvals"
        task_list = ["5-1","6-2","6-4"]
        t_max_episode_length = 1000
        experiment_parameters = [(seed, tgrace, task) for task in task_list for tgrace in [0.0, 0.05, 0.2, 0.5, 1.0] for seed in seeds]

        from progress_tracker import experimentProgressTracker
        tracker = experimentProgressTracker("supermario_tgraceexpdifferentvals", 0, len(experiment_parameters), min_exp_time=max_optimization_time*0.95)


        def run_with_experiment_index():
            idx = tracker.get_next_index()
            seed, tgrace, task = experiment_parameters[idx]
            real_tgrace = max(1,round(t_max_episode_length * tgrace))
            print(seed, tgrace, task)
            
            # Kill all old fceux processes
            subprocess.run("ps -eo pid,lstart,comm | grep fceux | sort -k 4,4 -k 5,5M -k 6,6n -k 7,7n | head -n -8 | awk '{print $1}' | xargs kill -9", shell=True, capture_output=False)
            time.sleep(0.5)
            print(f"Launching {task} with tgrace {tgrace} seed {seed} in supermario tgrace exp ...")
            res_filepath = f"results/data/tgrace_different_values/supermario{task}_{tgrace}_{seed}.txt"
            cmd = f"exec python3 other_RL/super-mario-neat/src/main.py train --gen 10000 --task {task} --seed {seed} --method {method} --resultfilename {res_filepath} --gracetime {real_tgrace} --experiment_index_for_log {idx} --max_optimization_time {max_optimization_time}"
            print(cmd)
            try:
                subprocess.run(cmd,shell=True, capture_output=False, timeout=max_optimization_time*1.2)
            except subprocess.TimeoutExpired:
                print("Break: subprocess timeout.")
                pass

        Parallel(n_jobs=parallel_threads, verbose=12)(delayed(run_with_experiment_index)() for _ in range(len(experiment_parameters)))
        exit(0)

    #endregion


    if sys.argv[1] == "--tgrace_nokill":
        import itertools
        import time

        seeds = list(range(2,32))
        max_optimization_time = 1400.0

        methods = ["tgraceexp"]
        task_list = ["5-1", "6-2", "6-4"]
        experiment_parameters = list(itertools.product(seeds, methods, task_list))


        from progress_tracker import experimentProgressTracker
        tracker = experimentProgressTracker("supermario_tgraceexpnokill", 0, len(experiment_parameters), min_exp_time=20.0)
        def run_with_experiment_index():
            idx = tracker.get_next_index()
            seed, method, task = experiment_parameters[idx]
            print(seed, method, task)

            # Kill all old fceux processes
            subprocess.run("ps -eo pid,lstart,comm | grep fceux | sort -k 4,4 -k 5,5M -k 6,6n -k 7,7n | head -n -8 | awk '{print $1}' | xargs kill -9", shell=True, capture_output=False)
            time.sleep(0.5)
            print(f"Launching {task} with seed {seed} in supermario tgrace exp ...")
            print(f"python3 other_RL/super-mario-neat/src/main.py train --gen {gens} --task {task} --seed {seed} --method {method} --resultfilename results/data/tgrace_experiment/supermario{task}_{seed}.txt --gracetime {gracetime} --experiment_index_for_log {idx}")
            subprocess.run(f"python3 other_RL/super-mario-neat/src/main.py train --gen {gens} --task {task} --seed {seed} --method {method} --resultfilename results/data/tgrace_experiment/supermario{task}_{seed}.txt --gracetime {gracetime} --experiment_index_for_log {idx} --max_optimization_time {max_optimization_time}",shell=True, capture_output=False)

        Parallel(n_jobs=parallel_threads, verbose=12)(delayed(run_with_experiment_index)() for _ in range(len(experiment_parameters)))
        exit(0)



    #endregion






    #region plot

    if sys.argv[1] == "--plot":
        import itertools
        import pandas as pd
        from matplotlib import pyplot as plt
        import numpy as np
        from scipy.stats import mannwhitneyu

        savefig_paths = ["results/figures/super_mario/", "/home/paran/Dropbox/BCAM/07_estancia_1/paper/images/super_mario/"]

        df_row_list = []
#        for fincrementsize in ("", "fincrementsize"):
        for fincrementsize in [""]:
            for method in methods:
                for seed in seeds:
                    i = 0
                    res_filepath = f"results/data/super_mario/task_{task}_{method}{fincrementsize}_{seed}.txt"
                    if exists(res_filepath):
                        with open(res_filepath, "r") as f:
                            all_text = f.readlines()
                            for line in all_text:
                                split_line = line.strip("\n").split(",")
                                fitness = float(split_line[1])
                                clock_time = float(split_line[2])
                                rw_time = float(split_line[3])
                                evals = int(split_line[4])
                                pe.add_data(task, method, seed, clock_time, evals)

                                maxevaltimes_each_controller = [float(el) for el in split_line[5].strip("()").split(";") if len(el) > 0]
                                if float(fitness) < -10e200:
                                    continue
                                df_row_list.append([seed, evals, rw_time, fitness, maxevaltimes_each_controller, clock_time, method, task, fincrementsize])
                                i += 1
                    print(i, "rows:", res_filepath)
        df_all = pd.DataFrame(df_row_list, columns=["seed", "evals", "rw_time", "fitness", "maxevaltimes_each_controller", "simulated_time", "method", "task", "fincrementsize"])

        import pandas as pd
        pd.set_option('display.max_rows', 1000)
  
        def get_xy_from_df(time_name_in_df, x_min, x_max, x_nsteps, df: pd.DataFrame):
            x = []
            y_median = []
            y_lower = []
            y_upper = []
            every_y = []

            for runtime in np.linspace(x_min, x_max, num=x_nsteps):
                if df[df[time_name_in_df] <= runtime].shape[0] <= 2:
                    continue
                
                fitnesses = []
                for seed in seeds:
                    f_with_seed_and_runtime_leq = df[(df[time_name_in_df]<=runtime) & (df["seed"]==seed)]["fitness"]
                    f_with_runtime_geq = df[df[time_name_in_df]>runtime]
                    if len(f_with_runtime_geq) < 4:
                        return x, y_median, y_lower, y_upper, every_y
                    if len(f_with_seed_and_runtime_leq) > 4:
                        fitnesses.append(max(f_with_seed_and_runtime_leq))
                    else:
                        fitnesses.append(0)
                x.append(runtime)
                every_y.append(fitnesses)
                y_median.append(np.quantile(np.array(fitnesses), 0.5))
                y_lower.append(np.quantile(np.array(fitnesses), 0.25))
                y_upper.append(np.quantile(np.array(fitnesses), 0.75))
            
            return x, y_median, y_lower, y_upper, every_y



        x_min = 0.0
        y_min = 5

        x_max = 50000.0
        x_nsteps = 200

        for fincrementsize in [""]:

            x_list = []
            y_median_list = []
            y_lower_list = []
            y_upper_list = []
            every_y_halve_list = []
            for method in methods:

                x, y_median, y_lower, y_upper, every_y_halve = get_xy_from_df("simulated_time", x_min, x_max, x_nsteps, df_all[(df_all["method"] == method) & (df_all["fincrementsize"] == fincrementsize)])
                x_list.append(x)
                y_median_list.append(y_median)
                y_lower_list.append(y_lower)
                y_upper_list.append(y_upper)
                every_y_halve_list.append(every_y_halve)


            def get_test_result(x, y, alpha = 0.05):
                x = x[0:min(len(x), len(y))]
                y = y[0:min(len(x), len(y))]
                if len(x) < 5:
                    print("WARNING: statistical test with less than 5 samples. Probably wont be significant.")
                return mannwhitneyu(x, y, alternative='two-sided')[1] < alpha


            # # This assertion is required for doing the tests. We are comparing the samples based on the samples
            # # in every_y_halve and every_y_constant. Consequently, the indexes in these samples need to correspond 
            # # to the same x values.
            # assert x_constant == x_halve             
            # test_results_true = np.where([get_test_result(every_y_halve[i], every_y_constant[i]) for i in range(len(every_y_halve))])[0]



            # import code
            # code.interact(local=locals())


            x_list = [[el/3600 for el in x] for x in x_list]
            x_max = x_max/3600
            plt.figure(figsize=(4, 3))
            for x, y_median, y_lower, y_upper, every_y_halve, method, method_name, color, marker in zip(x_list, y_median_list, y_lower_list, y_upper_list, every_y_halve_list, methods, method_plot_name_list, ["tab:blue", "tab:orange", "tab:green"], ["o","x",","] ):
                plt.plot(x, y_median, label=f"{method_name}" if apply_legend_to_first_plot else None, color=color, marker=marker, markevery=(0.2, 0.4))
                plt.fill_between(x, y_lower, y_upper, color=color, alpha=.25)
                if len(x) != 0:
                    x_max = min(x_max, max(x))
                else:
                    print(x)
                # plt.plot(np.array(x_halve)[test_results_true], np.repeat(y_min, len(test_results_true)), linestyle="None", marker = "_", color="black", label="$p < 0.05$")
                # plt.scatter(df_halve_maxevaltime["rw_time"], df_halve_maxevaltime["fitness"], marker="o", label = "halve runtime", alpha=0.5, color="red")

            best_f = df_all["fitness"].max()
            plt.plot((0, x_max), (best_f, best_f), color="gray", linestyle="--", label="best-found")
            plt.xlim((0, x_max))

            plt.annotate(f'Level: {task}', (0.05, 0.85), xycoords='axes fraction')  # Add level to each plot

            if apply_legend_to_first_plot:
                plt.legend()
                apply_legend_to_first_plot = False

            plt.minorticks_on()
            plt.xlabel("Optimization time in hours")
            plt.ylabel("Objective value")

            plt.subplots_adjust(top=0.96, bottom=0.02)
            plt.tight_layout(pad=0.0)

            for path in savefig_paths:
                plt.savefig(path + f"/task_{task}_{fincrementsize}_exp_line.pdf", bbox_inches='tight')
            plt.close()

        # Plot runtimes
        df = pd.read_csv("results/data/super_mario/runtimes.csv", header=None, names=["task", "method", "seed", "fitness", "time"])


        plt.figure()
        for i, method in enumerate(methods):
            sub_df = df[(df["task"] == task) & (df["method"] == method)]
            plt.scatter(sub_df["time"], sub_df["fitness"], color = ["red", "green", "blue"][i], marker=["o", ".", "x"][i], label=method)
        plt.legend()
        for path in savefig_paths:
            plt.savefig(path + f"/task_{task}_time_vs_fitness.pdf")
        plt.close()


        def generate_evaluation_ratio_plot():
            print("Generating evaluations/time plots")

            fig, ax = plt.subplots(figsize=(6, 3))
            linestyle_list=['-','-','-','-','-', '-','-',':',':']
            marker_list=   ['x','h','d','^',',', '.',"*",',', '.']
            color_list=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b','#008b8b','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

            for j, task in enumerate(task_list):
                quantiles, y = pe.get_proportion(task, "nokill", "bestasref") 
                ax.plot(quantiles, y, label=task, color=color_list[j], marker=marker_list[j], linestyle=linestyle_list[j])

            ax.set_xlabel(r"Normalized optimization runtime budget")
            ax.set_ylabel("Proportion of solutions evaluated")
            ax.set_ylim((1.0, ax.get_ylim()[1]))
            # Create a legend and place it to the right of the plot
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
            # Adjust layout
            fig.tight_layout()
            for path in savefig_paths:
                fig.savefig(path + f"/evals_proportion_supermario.pdf")


        if index == len(task_list)-1:
            generate_evaluation_ratio_plot()

    #endregion


    print("done.")
