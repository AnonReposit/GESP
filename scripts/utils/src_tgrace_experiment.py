import csv
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from typing import Callable, Any, Iterable, Tuple
from tqdm import tqdm as tqdm
from copy import deepcopy
from termcolor import colored
import time



MAX_TIME_TGRACE_DIFFERENT_VALUES_EXP=1200.0



def exit_after_k():
    exit_after_k.exit_call_counter += 1
    if exit_after_k.exit_call_counter == 2:
        exit(0)
exit_after_k.exit_call_counter = 0

def print_array_with_highlight(arr, color, position):
    assert position >= 0
    """
    Print the numpy array in a single line with values separated by spaces,
    highlighting the element at the specified position with the given color.

    Parameters:
    - arr (numpy.ndarray): The input numpy array.
    - color (str): The color for highlighting (e.g., 'red', 'green', 'blue', etc.).
    - position (int): The position of the element to be highlighted.

    Returns:
    - None
    """
    formatted_array = " ".join(map(str, arr))
    formatted_array = formatted_array.split()
    formatted_array[position] = colored(formatted_array[position], color)
    formatted_array = " ".join(formatted_array)
    print(f"[{formatted_array}]") 



class TgraceDifferentValuesLogger:
    def __init__(self, file_path:str, max_optimization_time:float, replace_existing:bool=False):
        assert isinstance(file_path, str)
        assert isinstance(max_optimization_time, float)
        assert isinstance(replace_existing, bool)


        # Create the directory if it doesn't exist
        log_dir = os.path.dirname(file_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Check if the log file already exists
        if os.path.exists(file_path):
            if replace_existing:
                os.remove(file_path)
            else:
                raise FileExistsError(f"The log file '{file_path}' already exists.")

        self.file_path = file_path
        self.row_count = 0  # Initialize row count

        # Open the file and create the CSV writer
        self.csvfile = open(self.file_path, 'a', newline='')
        self.writer = csv.writer(self.csvfile)
        self.f_best = -1e20
        self.header_written = False
        self.start_time = time.time()
        self.max_optimization_time = max_optimization_time

    def tic(self):
        self.start_time = time.time()
    
    def toc(self):
        return time.time() - self.start_time


    def log_values(self, f, step):
        t = self.toc()
        # print("log", t, self.f_best, f, self.row_count, step)

        if not self.header_written:
            # Write header row
            self.writer.writerow(["t","f","solution","step"])
            self.header_written = True

        if f > self.f_best:
            self.f_best = f
            self.writer.writerow([t, self.f_best, self.row_count, step])

        self.row_count += 1
        assert type(self.max_optimization_time) == float, f"self.T = {self.max_optimization_time} must be float, instead it was {type(self.max_optimization_time)}."
        if t > self.max_optimization_time:
            self.writer.writerow([t, self.f_best, self.row_count, step])
            self.csvfile.close()
            print("Done TgraceDifferentValuesLogger!")
            exit(0)


        



class TgraceNokillLogger:
    def __init__(self, file_path:str, max_optimization_time:float, replace_existing:bool=False, logevery:int=1):

        assert isinstance(file_path, str)
        assert isinstance(max_optimization_time, float)
        assert isinstance(replace_existing, bool)
        assert isinstance(logevery, int)

        self.max_optimization_time = max_optimization_time

        # Create the directory if it doesn't exist
        log_dir = os.path.dirname(file_path)
        self.logevery=logevery
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Check if the log file already exists
        if os.path.exists(file_path):
            if replace_existing:
                os.remove(file_path)
            else:
                raise FileExistsError(f"The log file '{file_path}' already exists.")

        self.file_path = file_path
        self.row_count = 0  # Initialize row count

        # Open the file and create the CSV writer
        self.csvfile = open(self.file_path, 'a', newline='')
        self.writer = csv.writer(self.csvfile)

        # self.header_written = False
        self.tic()


    def tic(self):
        self.start_time = time.time()
    
    def toc(self):
        return time.time() - self.start_time

    def log_values(self, values):

        time = self.toc()
        self.row_count += 1

        values = values[::self.logevery]
        self.writer.writerow([time] + list(values))

        if time > self.max_optimization_time:
            self.csvfile.close()
            print("Done TgraceNokillLogger!")
            exit(0)


class tgrace_exp_figures():

    @classmethod
    def _get_seed_from_filepath(self, filepath:str):
        return int(filepath.split("_")[-1].removesuffix(".txt"))

    @classmethod
    def _get_filepath_list(self, experiment_name, experiment_result_path):
        res = []
        filename_list = os.listdir(experiment_result_path)
        filename_list = filter(lambda x: ".txt" in x, filename_list)
        filename_list = sorted(filename_list, key=lambda x: self._get_seed_from_filepath(x))

        for filename in filename_list:
            if filename.startswith(experiment_name):
                file_path = os.path.join(experiment_result_path, filename)
                res.append(file_path)
        return res

    def _get_max_row_length(self, filename_list):
        largest_column_count = 0
        for filepath in tqdm(filename_list, desc="counting n-columns"):
            with open(filepath, 'r') as temp_f:
                lines = temp_f.readlines()
                for l in lines:
                    column_count = len(l.split(",")) + 1
                    largest_column_count = max(column_count, largest_column_count)
        return largest_column_count


    def __init__(self, experiment_name, plot_label, experiment_result_path):


        print("TODO: Reporting objective value when environment terminates the evaluation.")
        self.combined_df: pd.DataFrame = None
        self.plot_label = plot_label
        self.seed_list = []
        self.experiment_name = experiment_name
        filepath_list = self._get_filepath_list(experiment_name, experiment_result_path)
        self.largest_column_count = self._get_max_row_length(filepath_list)
        column_names = ["time"]+[i for i in range(0, self.largest_column_count-2)]

        for filepath in filepath_list:
            seed = self._get_seed_from_filepath(filepath)
            self.seed_list.append(seed)
            df = pd.read_csv(filepath, header=None, delimiter=",", names=column_names)
            df.insert(0, "seed", seed)
            self.combined_df = pd.concat([self.combined_df, df], axis=0).reset_index(drop=True)
        self.drop_seeds_with_low_time()
        self.reset_refs_stopping()
        self.ref_len = None




    def drop_seeds_with_low_time(self):
        prop_maxtime_drop = 0.9
        max_time_indices = self.combined_df.groupby('seed')['time'].idxmax()
        self.t_max = max(self.combined_df.loc[max_time_indices]["time"]) * prop_maxtime_drop
        max_time_rows = self.combined_df.loc[max_time_indices]["time"]


        print(f"Drop rows with a total time lower than {prop_maxtime_drop} * max_time.")
        filtered_series = max_time_rows[max_time_rows < self.t_max]
        indexes_below_0_9 = filtered_series.index.tolist()
        for idx_to_drop in indexes_below_0_9:
            seed = int(self.combined_df.loc[idx_to_drop]["seed"])
            self.combined_df = self.combined_df[self.combined_df['seed'] != seed]
        self.combined_df.reset_index(drop=True)
        self.seed_list = list(self.combined_df["seed"].unique())
        print(f"{len(indexes_below_0_9)} seeds where discarded, and a total of {len(self.seed_list)} valid seeds are left.")


    def reset_refs_stopping(self):
        self.gesp_refs = None
        self.gesp_current_steps = None
        self.gesp_current_steps_w_gesp = None
        self.gesp_current_best_f = None
        self.gesp_current_best_f_w_gesp = None


    def _get_ratio_where_gesp_eq_or_better(self):
        # It needs to be the value at the end. Otherwise the time graph makes no sense.
        w_gesp_f = self.gesp_current_best_f_w_gesp[-1]
        w_gesp_step = self.gesp_current_steps_w_gesp[-1]
        wo_gesp_f = None

        for step,f in zip(self.gesp_current_steps, self.gesp_current_best_f):
            if step > w_gesp_step:
                break
            wo_gesp_f = f
        assert wo_gesp_f != None
        # comp = lambda a, b: 0.5 if a == b else 1 if a > b else 0
        comp = lambda a, b: 1.0 if b >= a else 0.0
        return comp(wo_gesp_f, w_gesp_f) # with gesp equal or better -> 1.0 


    def when2stopGESP(self, f_array, t_grace_proportion):
        assert t_grace_proportion <= 1.0


        # Create copy where nans are -infty
        f_array_no_nans = np.copy(f_array)
        f_array_no_nans[np.where(np.isnan(f_array_no_nans))[0]] = np.float64(-1e10)

        # First call: fill self.gesp_refs 
        if self.gesp_refs is None:
            self.gesp_refs = np.copy(f_array_no_nans)

        # Get episode length
        max_episode_length = len(f_array)
        episode_length_wo_gesp = len(f_array) - len(np.where(np.isnan(f_array))[0])

        _grace_steps = round(t_grace_proportion * len(f_array))

        # print("np.max(",f_array[:max_episode_length-_grace_steps], "|",       f_array[_grace_steps:], ") < ")
        # print("np.max(",self.gesp_refs[:max_episode_length-_grace_steps], "|", self.gesp_refs[_grace_steps:], ")")
        _episode_length_w_gesp =  np.where(
            np.maximum(       f_array[:max_episode_length-_grace_steps],        f_array[_grace_steps:]) < 
            np.minimum(self.gesp_refs[:max_episode_length-_grace_steps], self.gesp_refs[_grace_steps:])
        )[0]
        episode_length_w_gesp = (_episode_length_w_gesp[0] + 1 + _grace_steps) if len(_episode_length_w_gesp) > 0 else episode_length_wo_gesp

        # print(episode_length_w_gesp)
        # print(episode_length_wo_gesp)
        # print("---")
        # exit_after_k()

        # Find why the evaluation stopped assuming gesp is applied.
        if episode_length_w_gesp == max_episode_length:
            why_evaluation_stopped = "maxepisodelength" # If episode evaluated until the end
        elif episode_length_w_gesp == episode_length_wo_gesp and episode_length_wo_gesp < max_episode_length: 
            why_evaluation_stopped = "problemspecific" # Evaluation stopped by environment (i.e mario died)
        elif episode_length_w_gesp < episode_length_wo_gesp:
            why_evaluation_stopped = "gesp" # Gesp early stopped the evaluation.
        else:
            raise ValueError(f"One of the previous three conditions should be True.\n episode_length_w_gesp = {episode_length_w_gesp}\n episode_length_wo_gesp = {episode_length_wo_gesp}\n max_episode_length = {max_episode_length}\n")

        is_monotone_increasing = False
        if self.experiment_name in (
            "garagegymCartPole-v1",
            "supermario5-1",
            "supermario6-2",
            "supermario6-4",
            "garagegymAnt-v3",
            "garagegymHopper-v3",
            # "garagegymPendulum-v1",
            # "garagegymHalfCheetah-v3",
            # "garagegymSwimmer-v3",
            # "veenstra",
             ):
            is_monotone_increasing = True


        assert is_monotone_increasing or (episode_length_wo_gesp == max_episode_length), f"The environment {self.experiment_name} needs to have either \n1) a monotone increasing f \nor\n2) A constant episode length."

        if is_monotone_increasing: 
            is_better_than_best_found_wo_gesp = np.max(f_array_no_nans) > np.max(self.gesp_refs)
        else:
            is_better_than_best_found_wo_gesp = f_array_no_nans[-1] > self.gesp_refs[-1]


        if is_better_than_best_found_wo_gesp:
            self.gesp_refs = f_array_no_nans[:]

        was_early_stopped = not (why_evaluation_stopped != "gesp")

        if self.gesp_current_steps is None:
            self.gesp_current_steps = [episode_length_wo_gesp]
            self.gesp_current_steps_w_gesp = [episode_length_w_gesp]
            self.gesp_current_best_f = [f_array[episode_length_wo_gesp-1]]
            self.gesp_current_best_f_w_gesp = [f_array[episode_length_w_gesp-1]]
        else:
            self.gesp_current_steps.append(       self.gesp_current_steps[-1]        + episode_length_wo_gesp)
            self.gesp_current_steps_w_gesp.append(self.gesp_current_steps_w_gesp[-1] + episode_length_w_gesp)
            self.gesp_current_best_f.append(       max(self.gesp_current_best_f[-1],        f_array[episode_length_wo_gesp-1]))
            self.gesp_current_best_f_w_gesp.append(max(self.gesp_current_best_f_w_gesp[-1], f_array[episode_length_w_gesp-1]))


        # print("max_episode_length",max_episode_length)
        # print("episode_length_wo_gesp",episode_length_wo_gesp)
        # print("episode_length_w_gesp",episode_length_w_gesp)
        # print_array_with_highlight(f_array,"yellow", episode_length_wo_gesp-1)
        # print("f_wo_gesp",f_array[episode_length_wo_gesp-1])
        # print_array_with_highlight(f_array,"yellow", episode_length_w_gesp-1)
        # print("f_w_gesp",f_array[episode_length_w_gesp-1])
        # print("why_evaluation_stopped",why_evaluation_stopped)
        # print("---")
        # input("Press Enter key to continue...")


        res = {
            "episode_length_w_gesp": self.gesp_current_steps_w_gesp[-1],
            "episode_length_wo_gesp": self.gesp_current_steps[-1],
            "is_better_than_best_found_wo_gesp": is_better_than_best_found_wo_gesp,
            "was_early_stopped": was_early_stopped,
        }

        return res



    def get_proportion_timesaved_bestsolsmised(self, when2stopfunc):
        """
        when2stopfunc: Given a array of f-values, it tells you the index in which the evaluation would be stopped. 
        """
        proportion_best_solution_not_missed_list = []
        proportion_frames_evaluated_list = []
        proportion_with_gesp_eq_or_better_list = []
        assert self.combined_df.shape[0] > 2, "Dataframe is empty or has only one row."
        for seed in self.seed_list:
            self.reset_refs_stopping()
            iterable_of_observed_f = [el[1][2:].to_numpy() for el in self.combined_df[self.combined_df['seed'] == seed].iterrows()]
            res_list = [when2stopfunc(row) for row in iterable_of_observed_f]
            best_solution_not_missed = [(res["is_better_than_best_found_wo_gesp"] and (not res["was_early_stopped"])) for res in res_list if res["is_better_than_best_found_wo_gesp"]]
            if len(best_solution_not_missed) == 0:
                print("is_better_than_best_found_wo_gesp was always false in res...")
                print("This means that the optimal objective function was found in the first evaluated solution and was not further improved, which makes no sense.")
                print("res=",res_list)
                raise ValueError("res[\"is_better_than_best_found_wo_gesp\"] was always false")
                return None, None, None
            proportion_best_solution_not_missed = np.mean(np.int16(best_solution_not_missed))
            proportion_best_solution_not_missed_list.append(proportion_best_solution_not_missed)
            proportion_frames_evaluated = res_list[-1]["episode_length_w_gesp"] / res_list[-1]["episode_length_wo_gesp"]
            proportion_frames_evaluated_list.append(proportion_frames_evaluated)
            proportion_with_gesp_eq_or_better_list.append(self._get_ratio_where_gesp_eq_or_better())
        return proportion_best_solution_not_missed_list, proportion_frames_evaluated_list, proportion_with_gesp_eq_or_better_list


    def plot_tgrace_param(self):
        
        x = np.linspace(0.0, 1.0, 101, endpoint=True)
        y_missed_median = np.zeros_like(x, dtype=np.float64)
        y_frames_median = np.zeros_like(x, dtype=np.float64)
        y_better_median = np.zeros_like(x, dtype=np.float64)

        for i, t_grace in tqdm(list(enumerate(x))):
            proportion_best_missed_list, proportion_frames_evaluated_list, proportion_with_gesp_worse_list = self.get_proportion_timesaved_bestsolsmised(lambda x: self.when2stopGESP(x, t_grace))
            if proportion_best_missed_list is None:
                print("Skipping plot for", env_name, "because applying GESP made no difference.")
                return
            x[i] = t_grace
            y_missed_median[i] = np.mean(proportion_best_missed_list)
            y_frames_median[i] = np.mean(proportion_frames_evaluated_list)
            y_better_median[i] = np.mean(proportion_with_gesp_worse_list)
        plt.figure(figsize=(4, 3) if plot_label == "classic: cart pole" else (4, 2.17))
        plt.ylim(0.0, 1.05)
        plt.plot(x, y_missed_median, linestyle="-", color="#1f77b4", label="best solution not missed")
        plt.plot(x, y_frames_median, linestyle="--", color="#ff7f0e", label="steps computed")
        plt.plot(x, y_better_median, linestyle="-.", color="#2ca02c", label="gesp improves result")

        plt.xlabel(None)

        if plot_label == "classic: cart pole":
            plt.subplots_adjust(top=0.6)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.8))

        def format_plot_label(plot_label: str):
            plot_label = plot_label.replace(" ", "\\ ")
            if ":" not in plot_label:
                return r"$\mathbf{"+plot_label+"}$"
            else:
                return r"$\mathbf{" + plot_label.split(":")[0] + "}$: " + r"$\mathit{" + plot_label.split(":")[1] + "}$"

        plt.text(0.5, 1.2, format_plot_label(self.plot_label), ha='center', va='top')
        plt.tight_layout()
        plt.savefig(f"results/figures/tgrace_experiment/{self.experiment_name}_proportion_average_no_runtime.pdf")
        plt.close()




def _tgrace_different_get_data(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        if len(lines) == 0:
            return None
        last_line = lines[-1].removesuffix("\n")
    last_line_list = last_line.split(",")

    # Retrieve the 'f' value from the row with the highest 't' value that's still less than 'max_time'
    max_time=MAX_TIME_TGRACE_DIFFERENT_VALUES_EXP
    df = pd.read_csv(file_path)
    f = df[df["t"] < max_time].iloc[-1]["f"]



    return {
        "t":float(last_line_list[0]),
        # "f":float(last_line_list[1]),
        "f":f,
        "sol_idx":int(last_line_list[2]),
        "step":int(last_line_list[3])
    }
    




def plot_tgrace_different_values(task_name, plot_label, experiment_result_path):
    path_list = tgrace_exp_figures._get_filepath_list(task_name,experiment_result_path)
    unique_tgrace_value_list = sorted(list(set(map(lambda x: x.split("_")[-2], path_list))))

    all_dfs = []
    for t_grace_value in unique_tgrace_value_list:
        n_resfile_with_no_data = 0
        file_path_list = [el for el in path_list if f"_{t_grace_value}_" in el]
        df_row_list = []
        for file_path in file_path_list:
            data_on_file = _tgrace_different_get_data(file_path)
            if data_on_file is None:
                n_resfile_with_no_data += 1
                continue
            df_row_list.append(pd.DataFrame([data_on_file]))
        df = pd.concat(df_row_list, ignore_index=True)
        out_of_range_rows = df[(df["t"] < 0.95 * df["t"].median()) | (df["t"] > 1.05 * df["t"].median())]
        # # Remove those rows from the dataframe.
        # df = df[(df["t"] >= 0.95 * df["t"].median()) & (df["t"] <= 1.05 * df["t"].median())]
        print(f"{len(out_of_range_rows)} rows are not within 95% and 105% of the median runtime.")
        print(f"{n_resfile_with_no_data} result files are empty.")

        all_dfs.append(df)




    import matplotlib.pyplot as plt
    import seaborn as sns


    # Plot f
    fig, axs = plt.subplots(1, len(all_dfs), sharey=True, figsize=(4, 2))
    for i, (df, t_grace_value) in enumerate(zip(all_dfs, unique_tgrace_value_list)):
        #sns.violinplot(y=df["f"], ax=axs[i], color=colors[i % len(colors)], inner="quartile")
        sns.boxplot(y=df["f"], ax=axs[i], color="white", linecolor="auto")
        axs[i].set_xlabel(f"{t_grace_value}")  # Set t_grace_value as x-tick label
        axs[i].set_ylabel("Objective value")
        axs[i].set_title("")  # Remove the title
        plt.setp(axs[i].artists, edgecolor = 'k', facecolor='w')
        plt.setp(axs[i].lines, color='k')
        average_f = df['f'].mean()
        axs[i].axhline(average_f, color='red', linestyle='--')


    def format_plot_label(plot_label: str):
        plot_label = plot_label.replace(" ", "\\ ")
        if ":" not in plot_label:
            return r"$\mathbf{"+plot_label+"}$"
        else:
            return r"$\mathbf{" + plot_label.split(":")[0] + "}$: " + r"$\mathit{" + plot_label.split(":")[1] + "}$"

    fig.text(0.5, 1.0, format_plot_label(plot_label), ha='center', va='top')
    plt.tight_layout()
    plt.savefig(f"results/figures/tgrace_different_values/f_violin_{task_name}.pdf")
    plt.close()

    # Plot steps per second.
    fig, axs = plt.subplots(1, len(all_dfs), sharey=True, figsize=(4, 2))
    for i, (df, t_grace_value) in enumerate(zip(all_dfs, unique_tgrace_value_list)):
        sns.violinplot(y=df["step"]/df["t"], ax=axs[i], color="white", inner="quartile")
        axs[i].set_xlabel(f"{t_grace_value}")  # Set t_grace_value as x-tick label
        axs[i].set_ylabel("steps per second")
        axs[i].set_title("")  # Remove the title
        # if task_name in ("veenstra"):
        #     axs[i].set_yscale("log")

    fig.text(0.5, 1.0, plot_label, ha='center', va='top')
    plt.tight_layout()
    plt.savefig(f"results/figures/tgrace_different_values/steps_per_second_violin_{task_name}.pdf")
    plt.close()


    # Plot n solutions.
    fig, axs = plt.subplots(1, len(all_dfs), sharey=True, figsize=(4, 2))
    for i, (df, t_grace_value) in enumerate(zip(all_dfs, unique_tgrace_value_list)):
        sns.violinplot(y=df["sol_idx"], ax=axs[i], color="white", inner="quartile")
        axs[i].set_xlabel(f"{t_grace_value}")  # Set t_grace_value as x-tick label
        axs[i].set_ylabel("n solutions evaluated")
        axs[i].set_title("")  # Remove the title

    fig.text(0.5, 1.0, plot_label, ha='center', va='top')
    plt.tight_layout()
    plt.savefig(f"results/figures/tgrace_different_values/n_solutions_violin_{task_name}.pdf")
    plt.close()



if __name__ == "__main__":
    # Call the function with default parameters

    for env_name in [
        "garagegymCartPole-v1",
        "garagegymPendulum-v1",
        "supermario5-1",
        "supermario6-2",
        "supermario6-4",
        "garagegymAnt-v3",
        "garagegymHopper-v3",
        "garagegymHalfCheetah-v3",
        "garagegymSwimmer-v3",
        "veenstra",
        ]:
        
        plot_label = {
            "garagegymCartPole-v1":"classic: cart pole",
            "garagegymPendulum-v1":"classic: pendulum",
            "supermario5-1":"super mario: level 5-1",
            "supermario6-2":"super mario: level 6-2",
            "supermario6-4":"super mario: level 6-4",
            "garagegymAnt-v3":"mujoco: ant",
            "garagegymHopper-v3":"mujoco: hopper",
            "garagegymHalfCheetah-v3":"mujoco: half cheetah",
            "garagegymSwimmer-v3":"mujoco: swimmer",
            "veenstra":"L-System",
        }[env_name]

        print(f"Generating tgrace different values plots for environment {env_name}...")
        plot_tgrace_different_values(env_name, plot_label, "results/data/tgrace_different_values/")

        print(f"Generating nokill plots for environment {env_name}...")
        exp = tgrace_exp_figures(env_name, plot_label, "results/data/tgrace_experiment/")
        exp.plot_tgrace_param()

