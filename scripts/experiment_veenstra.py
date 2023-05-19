from argparse import ArgumentError

from utils.UpdateParameter import *
import subprocess
import time
import re
from os.path import exists
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed
import argparse

# run 'chmod -R u+rw ~/Dropbox' to refresh dropbox
# import code; code.interact(local=locals()) # Start interactive mode for debug debugging


seeds = list(range(2,32))
parallel_threads = 7
gracetime = 130 # this is the runtime that the problem specific method allows for controllers in the first generation.

savefig_paths = ["results/figures"]

method_list = ["nokill", "bestasref", "problemspecific"]
method_plot_name_list = ["Standard", "GESP", "Problem Specific"]


if len(sys.argv) != 2:
    raise ArgumentError("this script requires only one argument --plot --launch_local")

if sys.argv[1] not in ("--plot", "--launch_local"):
    raise ArgumentError("this script requires only one argument --plot --launch_local")


#region local_launch

if sys.argv[1] == "--launch_local":
    import itertools
    import time


    def run_with_seed(seed):

        time.sleep(0.5)
        print(f"Launching with seed {seed} in experiment_halveruntime.py ...")
        for method in method_list:

            res_filepath = os.getcwd() + "/" + f"results/data/veenstra/{method}_{seed}.txt"
            bash_cmd = f"python3 other_RL/gym_rem2D/ModularER_2D/Demo2_Evolutionary_Run.py --method {method} --seed {seed} --gracetime {gracetime} --res_filepath {res_filepath}"
            print(bash_cmd)
            exec_res=subprocess.run(bash_cmd,shell=True, capture_output=True)
        
    #     run_with_seed_and_runtime(seed, "halving")
    Parallel(n_jobs=parallel_threads, verbose=12)(delayed(run_with_seed)(i) for i in seeds)


#endregion






#region plot

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



    import itertools
    import pandas as pd
    from matplotlib import pyplot as plt
    import numpy as np
    from scipy.stats import mannwhitneyu

    savefig_paths = ["results/figures/veenstra/", "/home/paran/Dropbox/BCAM/07_estancia_1/paper/images/veenstra/"]

    
    df_row_list = []
    df_row_list_test_scores = []
    for method in method_list:
        for seed in seeds:
            i = 0
            res_filepath = f"results/data/veenstra/{method}_{seed}.txt"
            if exists(res_filepath):
                with open(res_filepath, "r") as f:
                    all_text = f.readlines()
                    all_text, last_line = all_text[:-1], all_text[-1]
                    for line in all_text:
                        split_line = line.strip("\n").split(",")
                        fitness = float(split_line[1])
                        clock_time = float(split_line[2])
                        rw_time = float(split_line[3])
                        evals = int(split_line[4])
                        maxevaltimes_each_controller = [float(el) for el in split_line[5].strip("()").split(";") if len(el) > 0]
                        fintess_test = float(split_line[6])
                        pe.add_data("dummytaskname", method, seed, clock_time, evals)

                        if float(fitness) < -10e200:
                            continue
                        df_row_list.append([seed, evals, rw_time, fitness, maxevaltimes_each_controller, clock_time, method, fintess_test])
                        i += 1
            df_row_list_test_scores.append([seed, float(last_line), method])
            print(i, "rows:", res_filepath)
    df_all = pd.DataFrame(df_row_list, columns=["seed", "evals", "rw_time", "fitness", "maxevaltimes_each_controller", "simulated_time", "method", "fitness_test"])
    df_test_scores = pd.DataFrame(df_row_list_test_scores, columns=["seed","score","method"])
    import pandas as pd
    pd.set_option('display.max_rows', 1000)
    fig, ax = plt.subplots(figsize=(10,8))
    plt.suptitle('')
    df_test_scores.boxplot(column=['score'], by='method', ax=ax)
    for path in savefig_paths:
        plt.savefig(path + f"veenstra_final_boxplot.pdf")

    # import code; code.interact(local=locals()) # Start interactive mode for debug debugging
    
    for score_type in ["fitness","fitness_test"]:


        def get_xy_from_df(score_type, time_name_in_df, x_min, x_max, x_nsteps, df: pd.DataFrame):
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
                    f_with_seed_and_runtime_leq = df[(df[time_name_in_df]<=runtime) & (df["seed"]==seed)][score_type]
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
        y_min = -100

        x_max = 15000
        x_nsteps = 200




        x_list = []
        y_median_list = []
        y_lower_list = []
        y_upper_list = []
        every_y_halve_list = []
        for method in method_list:

            x, y_median, y_lower, y_upper, every_y_halve = get_xy_from_df(score_type, "simulated_time", x_min, x_max, x_nsteps, df_all[df_all["method"] == method])
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
        # # in every_y_halve and every_y_problemspecific. Consequently, the indexes in these samples need to correspond 
        # # to the same x values.
        # assert x_problemspecific == x_halve             
        # test_results_true = np.where([get_test_result(every_y_halve[i], every_y_problemspecific[i]) for i in range(len(every_y_halve))])[0]



        

        plt.figure(figsize=(4,3))
        plt.xlim((0, x_max / 3600))
        for x, y_median, y_lower, y_upper, every_y_halve, method, method_name, color, marker in zip(x_list, y_median_list, y_lower_list, y_upper_list, every_y_halve_list, method_list, method_plot_name_list, ["tab:blue", "tab:orange", "tab:green"], ["o","x",","]):
            plt.plot(np.array(x) / 3600, y_median, label=f"{method_name}", color=color, marker=marker, markevery=(0.2, 0.4))
            plt.fill_between(np.array(x) / 3600, y_lower, y_upper, color=color, alpha=.25)
            # plt.plot(np.array(x_halve)[test_results_true], np.repeat(y_min, len(test_results_true)), linestyle="None", marker = "_", color="black", label="$p < 0.05$")
            # plt.scatter(df_halve_maxevaltime["rw_time"], df_halve_maxevaltime["fitness"], marker="o", label = "halve runtime", alpha=0.5, color="red")
        plt.minorticks_on()
        plt.xlabel("Optimization time in hours")
        plt.ylabel("Objective value")
        plt.legend()
        plt.tight_layout()
        for path in savefig_paths:
            plt.savefig(path + f"veenstra_results_{score_type}.pdf")
        plt.close()



        print("Generating evaluations/time plots")

        def generate_evals_proportion_plot():
            fig, ax = plt.subplots(figsize=(4, 3))
            from cycler import cycler
            linestyle_list=['-','-','-','-','-','-',':',':',':']
            marker_list=['d','x','','^',',', '.','^',',', '.']
            color_list=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
            from matplotlib.lines import Line2D
            label_text=["GESP", "Problem Specific"]
            for j, method in enumerate(method_list[1:]):
                quantiles, y = pe.get_proportion("dummytaskname", method_list[0], method) 
                ax.plot(quantiles, y, label=label_text[j], color=color_list[j+1], marker=marker_list[j+1], linestyle=linestyle_list[j+1])

            fig.legend()
            ax.set_xlabel(r"Optimization time with respect to $t_{max}$")
            ax.set_ylabel("Proportion of solutions evaluated")
            ax.set_ylim((1.0, ax.get_ylim()[1]))
            ax.set_yscale("log")
            ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
            ax.grid(axis='y', color='0.95')
            fig.tight_layout()
            for path in savefig_paths:
                fig.savefig(path + f"/evals_proportion_VEENSTRA.pdf")

        # import code; code.interact(local=locals()) # Start interactive mode for debug debugging

        generate_evals_proportion_plot()






#endregion


print("done.")







