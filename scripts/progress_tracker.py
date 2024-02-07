import time
from pathlib import Path
import os
import fcntl
import pickle

def convert_from_seconds(seconds):


    mins, _ = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)

    # Format the result as a string
    result = f"{days} d, {hours} h, {mins} min"
    return result


class stopwatch:
    paused=False
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_t = time.time()
        self.pause_t=0
        self.paused=False

    def pause(self):
        self.pause_start = time.time()
        self.paused=True

    def resume(self):
        if self.paused:
            self.pause_t += time.time() - self.pause_start
            self.paused = False

    def get_time(self):
        # print("Pause time = ", self.pause_t)
        # print("Time without pause = ", time.time() - self.start_t)
        # print("Time = ", time.time() - self.start_t - self.pause_t)
        current_extra_pause_time = 0.0
        if self.paused:
            current_extra_pause_time = time.time() - self.pause_start
        return time.time() - self.start_t - self.pause_t - current_extra_pause_time

    def get_time_string_short_format(self):
        return "{:.4f}".format(self.get_time())

class Lock:
    def __init__(self, filenames):
        if isinstance(filenames, str):
            self.filenames = [filenames]
        self.files = [open(filename, 'r+') for filename in self.filenames]

    def __enter__(self):
        for file in self.files:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        if len(self.filenames) == 1:
            return self.files[0]
        else:
            return self.files

    def __exit__(self, exc_type, exc_value, traceback):
        for file in self.files:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            file.close()




class experimentProgressTracker:

    def __init__(self, progress_filename, start_index, max_index, min_exp_time = 0.0):

        self.min_exp_time = min_exp_time
        self.progress_filename, self.start_index, self.max_index = progress_filename, start_index, max_index

        self.start_ref = time.time()
        self.done = False
        
        path = Path('./'+ progress_filename)
        if not path.is_file() or os.stat(path).st_size < 4:
            with open(progress_filename,"a") as f:
                print("idx,done", file=f)
        self._clean_unfinished_jobs_from_log()
        self._save_to_file()

    def _save_to_file(self):
        with open(self.progress_filename + "_state.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def mark_index_done_external(progress_filename, idx):
        tmp_tracker = experimentProgressTracker._load_from_file(progress_filename)
        tmp_tracker.mark_index_done(idx)

    @staticmethod
    def _load_from_file(progress_filename: str) -> 'experimentProgressTracker':
        state_filename = progress_filename + "_state.pkl"
        if os.path.isfile(state_filename):
            with open(state_filename, "rb") as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Progress tracker state file '{state_filename}' not found.")

    def __repr__(self):
        return (
            f"experimentProgressTracker("
            f"\n  progress_filename='{self.progress_filename}',"
            f"\n  start_index={self.start_index},"
            f"\n  max_index={self.max_index},"
            f"\n  min_exp_time={self.min_exp_time},"
            f"\n  done={self.done}"
            f"\n)"
        )

    def _clean_unfinished_jobs_from_log(self):
        with Lock(self.progress_filename) as f:
            lines = f.readlines()
            f.seek(0)
            processed_lines = [line for line in lines if line.endswith(',1\n')]
            self.n_experiments_done_initially = len(processed_lines)
            f.writelines(["idx,done\n"]+processed_lines)
            f.truncate()

    def _get_next_index(self):
        with Lock(self.progress_filename) as f:
            content = f.read()
            for i in range(self.start_index, self.max_index+1):
                if f"{i}," not in content:
                    print(f"{i},0,{time.time()}", file=f, flush=True) # Mark experiment index in progress
                    return i
        return None

    def get_next_index(self):
        i = self._get_next_index()
        if i==None:
            self.done = True
            print("No more experiments left.")
            exit(0)

        print("------------\nWorking on experiment",i,"\n--------------")
        return i
    
    def mark_index_done(self, i):
        if self.done:
            exit(0)

        with Lock(self.progress_filename) as f:
            lines = []
            lines = f.readlines()

            def _index_with_substring(lst, substring):
                for i, item in enumerate(lst):
                    if substring in item:
                        return i
                raise ValueError(f"'{substring}' is not in list")


            index = _index_with_substring(lines, f"{i},0")

            def removesuffix(str_w_suffix, suffix):
                res = str_w_suffix
                while len(res) > 0 and res[-1] == suffix:
                    res = res[:-1]
                return res

            exp_start_time = float(removesuffix(lines[index].split(",")[-1], "\n"))
            assert time.time() - exp_start_time > self.min_exp_time

            lines[index] = f"{i},1\n"
            f.seek(0)
            f.truncate()
            f.writelines(lines)
            n_experiments_done_total = len([0 for line in lines if line.endswith(',1\n')])
            n_experiments_done_this_session = n_experiments_done_total - self.n_experiments_done_initially
            n_experiments_left = self.max_index - n_experiments_done_total - self.start_index
            elapsed_time = time.time() - self.start_ref
            time_left = elapsed_time / n_experiments_done_this_session * n_experiments_left

            print(f"{i},{n_experiments_left},{convert_from_seconds(time_left)}, {convert_from_seconds(elapsed_time)}\n")
            with open(self.progress_filename+"_log.txt","a") as f_log:
                f_log.write(f"{i},{n_experiments_left},{convert_from_seconds(time_left)}, {convert_from_seconds(elapsed_time)}\n")
