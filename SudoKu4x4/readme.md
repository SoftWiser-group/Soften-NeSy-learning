### Dataset
run `data_gen.py` to generate dataset by using MNIST 


### Repository contents

| file           | description                                                  |
| -------------- | ------------------------------------------------------------ |
| `rl_training.py` | implements the crude-RL strategy and its memory-augmented version. |
| `mcmc_training.py` | implements the softned nesy training with the crude MCMC (without projection). |
| `sl_training.py` | implements the semantic loss of a stochastic version (see Prop.2 in the paper). |
| `fs_training.py` | implements our stage I training, i.e., softened nesy training with projected MCMC. |
| `bs_training.py` | implements our stage II training, i.e., semi-supervised training with pseudo label (see App.B1 in the paper). |


