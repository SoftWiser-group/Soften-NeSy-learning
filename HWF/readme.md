### Dataset
Download data from https://github.com/liqing-ustc/NGS

### Repository contents

| file           | description                                                  |
| -------------- | ------------------------------------------------------------ |
| `rl_training.py` | implements the crude-RL strategy and its memory-augmented version. |
| `nsg_training.py` | implements the Neural-Grammer-Symbolic method proposed by Li et al.|
| `sl_training.py` | implements the semantic loss of a stochastic version (see Prop.2 in the paper). |
| `fs_training.py` | implements our stage I training, i.e., softened nesy training with projected MCMC. |
| `bs_training.py` | implements our stage II training, i.e., semi-supervised training with pseudo label (see App.B1 in the paper). |


