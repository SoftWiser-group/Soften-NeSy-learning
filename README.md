# Soften-NeSy-learning
Code for paper "Softened Symbol Grounding for Neuro-symbolic Systems" (ICLR 2023)

### Requirements
```
numpy
pytorch
networkx
joblib
z3-solver
cython
```

### Usage

For each task, first use the following command to start the Stage-I training. 

> python fs_training.py --seed 0 --exp_name fs --cooling_strategy 'log' --T 1.0 

and then use the following command to use the Stage-II training. 

> python3 bs_training.py --seed 0 --exp_name bs_log --net 'checkpoint/lenet_0_fs_log_best_0.t7'

Note that 

- The parameter T is actually the temperature gamma in the paper;
- We provide three cooling strategies, i.e., 'log', 'exp', and 'linear'. 

To reproduce the experimental results, Run the command `sh run.sh`



