from argparse import ArgumentError
from statistics import median
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils.UpdateParameter import *
import subprocess
import time
import re
from os.path import exists
import sys
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed
import argparse

# run 'chmod -R u+rw ~/Dropbox' to refresh dropbox


gens = 200
seeds = list(range(2,32))
parallel_threads = 7


savefig_paths = ["results/figures/garage_gym/", "../paper/images/garage_gym/"]

method_list = ["constant", "bestasref"]
method_plot_name_list = ["Standard", "GESP"]


# DTU = DisableTerminateUnhealthy
gymEnvName_list =         ['CartPole-v1',  'Pendulum-v1',  'HalfCheetah-v3',  'InvertedDoublePendulum-v2',  'Swimmer-v3', 'Hopper-v3' , 'Ant-v3'    , 'Walker2d-v3', 'Hopper-v3_DTU' , 'Ant-v3_DTU'    , 'Walker2d-v3_DTU']
plot_task_name_list =     ['cart pole'  , 'pendulum'    ,  'half cheetah'  , 'inverted double pendulum'  , 'swimmer'    , 'hopper'    ,        'ant', 'walker2d'   , 'hopper'    , 'ant'       , 'walker2d']
action_space_list =       ["discrete"   ,   "continuous",  "continuous"    ,  "continuous"               ,  "continuous", "continuous", "continuous", "continuous" , "continuous"    , "continuous"    , "continuous"     ]
max_episode_length_list = [          400,            200,  1000            ,                         1000,  1000        ,         1000,    1000     ,  1000        ,         1000    ,    1000         ,  1000            ]
plot_x_max_list =         [          100,           1000,   4500           ,                          800,  5500        ,         3500,    3000     ,  5000        ,         3500    ,    17000         ,  7000            ]


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




if sys.argv[1] == "--tgrace_different_values":
    import itertools
    import time

    method = "tgraceexpdifferentvals"
    seeds = list(range(2,32))
    task_list = ['CartPole-v1',
                 'Pendulum-v1',
                 'HalfCheetah-v3',
                 'InvertedDoublePendulum-v2',
                 'Swimmer-v3',
                 'Hopper-v3',
                 'Ant-v3',
                 'Walker2d-v3']


    experiment_parameters = [(seed, tgrace, task) for task in task_list for tgrace in [0.0, 0.05, 0.2, 0.5, 1.0] for seed in seeds]

    from progress_tracker import experimentProgressTracker
    tracker = experimentProgressTracker("garagegym_tgraceexpdifferentvals", 0, len(experiment_parameters), min_exp_time=20.0)
    gens = 100000

    def run_with_experiment_index():
        idx = tracker.get_next_index()
        seed, tgrace, task = experiment_parameters[idx]

        task_idx = task_list.index(task)
        action_space = action_space_list[task_idx]
        t_max_episode_length = max_episode_length_list[task_idx]
        max_optimization_time = plot_x_max_list[task_idx]


        real_tgrace = max(1,round(t_max_episode_length * tgrace))
        print(seed, tgrace, task)
        
        time.sleep(0.5)
        print(f"Launching {task} with tgrace {tgrace} seed {seed} in garagegym tgrace exp ...")
        res_filepath = f"results/data/tgrace_different_values/garagegym{task}_{tgrace}_{seed}.txt"
        cmd = f"python3 other_RL/meta_world_and_garage/test_example_garage_cart_pole_CMA_ES.py --method {method} --gymEnvName {task} --action_space {action_space} --seed {seed} --gracetime {real_tgrace} --gens {gens} --max_episode_length {t_max_episode_length} --res_filepath {res_filepath} --max_optimization_time {max_optimization_time}"
        print(cmd)
        try:
            subprocess.run(cmd,shell=True, capture_output=False, timeout=max_optimization_time*1.2+200.0)
            time.sleep(2.0)
            tracker.mark_index_done(idx)

        except subprocess.TimeoutExpired:
            print("Break: subprocess timeout.")
            pass

    Parallel(n_jobs=parallel_threads, verbose=12)(delayed(run_with_experiment_index)() for _ in range(len(experiment_parameters)))
    exit(0)



if sys.argv[1] == "--tgrace_nokill":
    import itertools
    import time

    method = "tgraceexp"
    seeds = list(range(2,32))
    task_list = ['CartPole-v1',
                 'Pendulum-v1',
                 'HalfCheetah-v3',
                 'InvertedDoublePendulum-v2',
                 'Swimmer-v3',
                 'Hopper-v3',
                 'Ant-v3',
                 'Walker2d-v3']


    experiment_parameters = [(seed, tgrace, task) for task in task_list for tgrace in [0.0, 0.05, 0.2, 0.5, 1.0] for seed in seeds]

    from progress_tracker import experimentProgressTracker
    tracker = experimentProgressTracker("garagegym_tgraceexpnokill", 0, len(experiment_parameters), min_exp_time=10.0)
    gens = 100000

    def run_with_experiment_index():
        idx = tracker.get_next_index()
        seed, tgrace, task = experiment_parameters[idx]

        task_idx = task_list.index(task)
        action_space = action_space_list[task_idx]
        t_max_episode_length = max_episode_length_list[task_idx]
        max_optimization_time = plot_x_max_list[task_idx]


        real_tgrace = max(1,round(t_max_episode_length * tgrace))
        print(seed, tgrace, task)
        
        time.sleep(0.5)
        print(f"Launching {task} with seed {seed} in garagegym tgrace exp ...")
        res_filepath = f"results/data/tgrace_experiment/garagegym{task}_{seed}.txt"
        cmd = f"python3 other_RL/meta_world_and_garage/test_example_garage_cart_pole_CMA_ES.py --method {method} --gymEnvName {task} --action_space {action_space} --seed {seed} --gracetime -1 --gens {gens} --max_episode_length {t_max_episode_length} --res_filepath {res_filepath} --max_optimization_time {max_optimization_time}"
        print(cmd)
        try:
            subprocess.run(cmd,shell=True, capture_output=False, timeout=max_optimization_time*1.2+200.0)
            time.sleep(2.0)
            tracker.mark_index_done(idx)

        except subprocess.TimeoutExpired:
            print("Break: subprocess timeout.")
            pass

    Parallel(n_jobs=parallel_threads, verbose=12)(delayed(run_with_experiment_index)() for _ in range(len(experiment_parameters)))
    exit(0)





for index, gymEnvName, action_space, max_episode_length, x_max in zip(range(len(gymEnvName_list)), gymEnvName_list, action_space_list, max_episode_length_list, plot_x_max_list):

    gracetime = round(max_episode_length * 0.2)
    plot_task_name = plot_task_name_list[index]

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

                res_filepath = f"results/data/garage_gym/gymEnvName_{gymEnvName}_{method}_{seed}.txt"
                bash_cmd = f"python3 other_RL/meta_world_and_garage/test_example_garage_cart_pole_CMA_ES.py --method {method} --gymEnvName {gymEnvName} --action_space={action_space} --seed {seed} --gracetime {gracetime} --gens {gens} --max_episode_length {max_episode_length} --res_filepath {res_filepath}"
                print(bash_cmd)
                exec_res=subprocess.run(bash_cmd,shell=True, capture_output=True)
            
        #     run_with_seed_and_runtime(seed, "halving")
        Parallel(n_jobs=parallel_threads, verbose=12)(delayed(run_with_seed)(i) for i in seeds)


    #endregion


    



    #region plot

    if sys.argv[1] == "--plot":
        import itertools
        import pandas as pd
        from matplotlib import pyplot as plt
        import numpy as np
        from scipy.stats import mannwhitneyu


        
        df_row_list = []
        for method in method_list:
            for seed in seeds:
                i = 0
                res_filepath = f"results/data/garage_gym/gymEnvName_{gymEnvName}_{method}_{seed}.txt"
                if exists(res_filepath):
                    with open(res_filepath, "r") as f:
                        all_text = f.readlines()
                        for line in all_text:
                            split_line = line.strip("\n").split(",")
                            fitness = float(split_line[1])
                            clock_time = float(split_line[2])
                            rw_time = float(split_line[3])
                            evals = int(split_line[4])
                            maxevaltimes_each_controller = [float(el) for el in split_line[5].strip("()").split(";") if len(el) > 0]
                            if float(fitness) < -10e200:
                                continue
                            df_row_list.append([seed, evals, rw_time, fitness, maxevaltimes_each_controller, clock_time, method, gymEnvName])
                            pe.add_data(gymEnvName, method, seed, clock_time, evals)
                            i += 1
                print(i, "rows:", res_filepath)
        df_all = pd.DataFrame(df_row_list, columns=["seed", "evals", "rw_time", "fitness", "maxevaltimes_each_controller", "simulated_time", "method", "gymEnvName"])

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
                    if len(f_with_seed_and_runtime_leq) >= 1:
                        fitnesses.append(max(f_with_seed_and_runtime_leq))
                    else:
                        continue

                if gymEnvName == "Ant-v3_DTU":
                    if len(fitnesses) < 6:
                        continue
                elif len(fitnesses) < len(seeds):
                    continue
                
                x.append(runtime)
                every_y.append(fitnesses)
                y_median.append(np.quantile(np.array(fitnesses), 0.5))
                y_lower.append(np.quantile(np.array(fitnesses), 0.25))
                y_upper.append(np.quantile(np.array(fitnesses), 0.75))
            
            return x, y_median, y_lower, y_upper, every_y



        x_min = 0.0

        x_max_suggested = 10e10
        for method in method_list:
            x_max_value_list = []
            sub_f = df_all[df_all["method"] == method]
            for seed in seeds:
                x_max_value_list.append(max(sub_f[sub_f["seed"] == seed]["simulated_time"]))

            x_max_suggested = min(x_max_suggested, median(x_max_value_list))

        print("x_max_suggested:", x_max_suggested)
        

        x_nsteps = 1000

        statistical_test_alpha = 0.01

        x_list = []
        y_median_list = []
        y_lower_list = []
        y_upper_list = []
        every_y_halve_list = []
        for method in method_list:

            x, y_median, y_lower, y_upper, every_y_halve = get_xy_from_df("simulated_time", x_min, x_max, x_nsteps, df_all[df_all["method"] == method])
            x_list.append(x)
            y_median_list.append(y_median)
            y_lower_list.append(y_lower)
            y_upper_list.append(y_upper)
            every_y_halve_list.append(every_y_halve)


        def get_test_result(x, y, alpha):
            x = x[0:min(len(x), len(y))]
            y = y[0:min(len(x), len(y))]
            if len(x) < 5:
                print("WARNING: statistical test with less than 5 samples. Probably wont be significant.")
            return mannwhitneyu(x, y, alternative='two-sided')[1] < alpha


        # # This assertion is required for doing the tests. We are comparing the samples based on the samples
        # # in every_y_halve and every_y_constant. Consequently, the indexes in these samples need to correspond 
        # # to the same x values.

        x_max_of_the_lowest_for_test = max((min(el) for el in x_list))
        index_constant_max_of_lowest = x_list[0].index(x_max_of_the_lowest_for_test)
        index_bestasref_max_of_lowest = x_list[1].index(x_max_of_the_lowest_for_test)


        y_constant_test  = every_y_halve_list[0][index_constant_max_of_lowest:]
        y_bestasref_test = every_y_halve_list[1][index_bestasref_max_of_lowest:]
        x_test = x_list[0][index_constant_max_of_lowest:]


        assert len(y_constant_test) == len(y_bestasref_test) == len(x_test)


        test_results_true = np.where([get_test_result(y_constant_test[i], y_bestasref_test[i], statistical_test_alpha) for i in range(len(y_constant_test))])[0]



        # import code
        # code.interact(local=locals())



        plt.figure(figsize=(4, 3))
        plt.xlim((0, x_max))
        for x, y_median, y_lower, y_upper, every_y_halve, method, method_name, color, marker in zip(x_list, y_median_list, y_lower_list, y_upper_list, every_y_halve_list, method_list, method_plot_name_list, ["tab:blue", "tab:orange", "tab:green"], ["o","x",","]):
            if gymEnvName=="Pendulum-v1":
                plt.yscale("symlog")
                # y_median = -np.array(y_median)
                # y_lower, y_upper = -np.array(y_upper), -np.array(y_lower)
            plt.plot(x, y_median, label=f"{method_name}", color=color, marker=marker, markevery=(0.2, 0.4))
            plt.fill_between(x, y_lower, y_upper, color=color, alpha=.25)
        y_min = plt.gca().get_ylim()[0]
        if x_max > 10000:
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 3))
        plt.plot(np.array(x_test)[test_results_true], np.repeat(y_min, len(test_results_true)), linestyle="None", marker = "_", color="black", label=f"$p < {statistical_test_alpha}$")
        plt.minorticks_on()
        plt.xlabel("Optimization time in seconds")
        plt.ylabel("Objective value")

        # plt.scatter(df_halve_maxevaltime["rw_time"], df_halve_maxevaltime["fitness"], marker="o", label = "halve runtime", alpha=0.5, color="red")


        if "DTU" not in gymEnvName:
            best_f = df_all["fitness"].max()
            plt.plot((0, x_max), (best_f, best_f), color="gray", linestyle="--", label="best-known")        
            # # To generate example on why bk makes no sense for this study.
            # plt.plot((0, x_max), (15000, 15000), color="black", linestyle="--", label="best-known")        


        plt.annotate(f'task: {plot_task_name}', (0.1, 0.15) if "DTU" in gymEnvName else (0.05, 0.85), xycoords='axes fraction')  # Add level to each plot

        if gymEnvName in ('Ant-v3', 'Ant-v3_DTU', 'CartPole-v1'):
            plt.legend()

            plt.subplots_adjust(top=0.96, bottom=0.02)
            plt.tight_layout(pad=0.0)
        for path in savefig_paths:
            # # To generate example on why bk makes no sense for this study.
            # if "Ant" in gymEnvName: 
            #    plt.savefig(path + f"/gymEnvName_{gymEnvName}_exp_line_best_known.pdf")
            plt.savefig(path + f"/gymEnvName_{gymEnvName}_exp_line.pdf", bbox_inches='tight')

        plt.close()

        if index == len(gymEnvName_list)-1:
            print("Generating evaluations/time plots")

            def generate_evals_proportion_plot():
                linestyle_list=['-','-','-','-','-','-',':',':',':']
                marker_list=['x','h','d','^',',', '.','^',',', '.']
                color_list=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
                
                for classic, reduced_task_list in zip([True, False], [gymEnvName_list[:2], gymEnvName_list[2:]]):
                    fig, ax = plt.subplots(figsize=(4, 3) if classic else (8,3))
                    label_text = [el.replace("-v3","").replace("-v2","").replace("_"," ").replace("-v1","") for el in reduced_task_list]
                    
                    def camel_case_to_lower_with_spaces(input_string):
                        words = re.findall(r'[a-z]+|[A-Z][a-z0-9]*', input_string)
                        result = ' '.join(words).lower()                    
                        return result
                    
                    label_text = [camel_case_to_lower_with_spaces(el) for el in label_text]
                    


                    for j, task in enumerate(reduced_task_list):
                        quantiles, y = pe.get_proportion(task, "constant", "bestasref") 
                        ax.plot(quantiles, y, label=label_text[j] if "d t u" not in label_text[j] else None, color=color_list[j], marker=marker_list[j], linestyle=linestyle_list[j])

                    
                    if classic:
                        fig.legend(loc='center')
                    else:
                        ax.plot([], [], color="black", linewidth=2, linestyle=":", label="Stop unhealthy disabled")                
                        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
                        plt.subplots_adjust(top=0.85)


                    ax.set_xlabel(r"normalized optimization runtime budget")
                    ax.set_ylabel("Proportion of solutions evaluated")
                    ax.set_ylim((0.5, ax.get_ylim()[1]))
                    plt.tight_layout()
                    for path in savefig_paths:
                        fig.savefig(path + f"/evals_proportion_classic_{classic}.pdf", bbox_inches="tight")
                


            # import code; code.interact(local=locals()) # Start interactive mode for debug debugging

            generate_evals_proportion_plot()






    #endregion


    print("done.")







