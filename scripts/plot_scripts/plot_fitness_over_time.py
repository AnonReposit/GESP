import sys
sys.path.insert(0, 'scripts/utils')
from functionsForcsvPlots import *


fig_dir = "results/figures"

# Results from new e-mail
figure_name = "FINTESS_TIME_Leni_nipes_logs"
dir_with_all_csv = "logs/nipes_15_5_14-57-31-1814-1756362047"

# # Executed with parameters from new e-mail
# figure_name = "FINTESS_TIME_executed_by_me"
# dir_with_all_csv = "logs/nipes_9_1_10-54-50-58-1859741572"

# # # Executed with OG codebase
# figure_name = "FITNESS_TIME_og_codebase"
# dir_with_all_csv = "logs/nipes_9_1_12-42-9-9928-3753403340"

neval_list, _, best_fit_list, avg_fit_list, pop_size_list = load_all_fitnesses(dir_with_all_csv + "/fitnesses.csv", 0)
_, eval_time_list = load_eval_time(dir_with_all_csv + "/eval_durations.csv")

plt.plot(neval_list, best_fit_list, label="fitness")
plt.legend()
plt.xlabel("Evaluations")
plt.ylabel("Objective function")
plt.title(figure_name)
plt.savefig(fig_dir + f"/{figure_name}.pdf")
