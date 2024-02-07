# Generalized Early Stopping for Episodic Policy Learning

Lengthy evaluation times are common in many optimization problems such as policy learning tasks, especially when they involve the physical world. When evaluating solutions, it is sometimes clear that the objective function is not going to increase with additional computation time (for example when a two wheel robot continuously spins in place).
In those cases, it makes sense to stop the evaluation early to save computation time. However, these approaches are problem specific and need to be specifically designed for the task at hand.


In our paper, we propose an early stopping method for policy learning. The proposed method only looks at the objective value at each time step and requires no problem specific knowledge. We tested the introduced stopping criterion in 6 policy learning environments and showed that it usually saves a significant amount of time. We also compare it with problem specific stopping criteria and show that it performs comparably, while being more general (it is not task specific).


## Usage

**Using the proposed stopping criterion does not require you to install anything.** It is quite easy to integrate the proposed early stopping criterion in your project. Each time you evaluate a solution (e.g. a policy) $\theta$, you get an objective value $f\[t\](\theta)$ at each time step (e.g. in RL it would be the cumulative reward of $\theta$ at time step $t$). Then you stop evaluating $\theta$ at time step $t$ if:

$$t > t_{grace}$$

**and**

$$\max \lbrace f\[t\](\theta), f\[t-t_{grace}\](\theta) \rbrace < \min \lbrace f\[t\](\theta_{best}), f\[t-t_{grace}\](\theta_{best})\rbrace$$

where $t_{grace}$ is set to a fraction of the maximum episode length $t_{max}$. As a rule of thumb, we propose $t_{grace} = 0.2 \cdot t_{max}$. We assume a maximization setting: a higher value of $f$ is a better value.


## Reproducing the experiments in the paper

### Installation

Clone the repo and check the instructions to install in the folders inside `other_RL/`. This step can be omitted for generating the plots, but it is necessary to perform this step to launch the experiments. You also need to install the python dependencies specified in `requirements.txt`. The `simulatedARE` experiment has additional dependencies (see `install_script_gecco21_leni_ARE.sh`).

### Usage

The experiments can be executed with the following four scripts:
- `scripts/experiment_garage_CMAES_gym.py`
- `scripts/experiment_simulatedARE.py`
- `scripts/experiment_supermario.py`
- `scripts/experiment_veenstra.py`

An extra parameter is required. 
- `--launch_local` to execute the experiments where GESP is compared to problem specific approaches or to using no early stopping. Generates the data for all the experimental result figures up to and including Figure 12.
- `--tgrace_different_values` to execute the experiments that compare different `t_grace` parameter values. Generates the data for Figure 13 in the paper.
- `--tgrace_nokill` to execute the experiments that estimate the success rate for different `t_grace` parameter values. Generates the data for Figure 14 in the paper.
- `--plot` to generate the figures in the paper up to Figure 12, from the results in `/results/data`. 

e.g. to generate the plots in the garage framework, 

```
python scripts/experiment_garage_CMAES_gym.py --plot
```


To generate Figure 13 and Figure 14 (the figures of the `t_grace` experiments) for all of the environments, use

```
python scripts/utils/src_tgrace_experiment.py
```



### Credits

List of external resources used in the project:
- [Coppelia Robotics](https://www.coppeliarobotics.com/)
- [Gym Rem2D by FrankVeenstra](https://github.com/FrankVeenstra/gym_rem2D)
- [Autonomous Robotics Evolution by Leni Le Goff et al.](https://bitbucket.org/autonomousroboticsevolution/evorl_gecco_2021/src/master/)
- [Super Mario by Vivek Verma](https://github.com/vivek3141/super-mario-neat)
- [Gym Garage](https://github.com/rlworkgroup/garage)
- [OpenAI Gym](https://www.gymlibrary.dev/)
- [MuJoCo](https://mujoco.org/)