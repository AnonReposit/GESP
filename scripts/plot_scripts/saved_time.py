import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


savefigpath = "/home/paran/Dropbox/BCAM/07_estancia_1/code/results/figures/"


def load_bw_theme(ax: plt.Axes):
    # taken from http://olsgaard.dk/monochrome-black-white-plots-in-matplotlib.html
    # Create cycler object. Use any styling from above you please
    monochrome = (cycler('color', ['k']) * cycler('marker', [' ', '.', 'x', '^']) * cycler('linestyle', ['-', '--', ':', '-.']))
    ax.set_prop_cycle(monochrome)
    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

k_values = [-100, -6, -4, -2, 0, 2, 4, 6]
fig, ax = plt.subplots(1,1)
load_bw_theme(ax)
for i,k in enumerate(k_values):
    x = np.linspace(0,1,1000)
    y = np.power(x, np.power(2.0,k))
    ax.plot(x,y, markevery=0.2, label="$k = "+str(k if k !=-100 else r'-\infty') + "$")
plt.legend()
plt.xlabel("$t$")
plt.ylabel("Adjusted runtime")
plt.savefig(savefigpath + "runtime_at_each_time_point.pdf")
plt.close()



k = np.linspace(-6, 6, 32)
time_saved_single_integral = np.power(np.power(2,k) + 1, -1)
time_saved_double_integral = np.power(np.power(2,k) + 1, -2)

fig, ax = plt.subplots(1,1)
load_bw_theme(ax)
ax.plot(k,time_saved_single_integral, markevery=0.2, label="Single time reduction")
ax.plot(k,time_saved_double_integral, markevery=0.2, label="Double time reduction")
ax.set_ylim((0,1))
plt.legend()
plt.xlabel("$k$")
plt.ylabel("Total runtime")
plt.savefig(savefigpath + "total_saved_time_depending_on_k.pdf")
plt.close()
