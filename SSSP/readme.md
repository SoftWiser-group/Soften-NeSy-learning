### Note 
First run the following command to compile cython code. 
> python3 setup.py build_ext --inplace 

Then, generate the dataset by `data_gen.py`.

### Repository contents

| file           | description                                                  |
| -------------- | ------------------------------------------------------------ |
| `data_gen.py` | generate the dataset by networkX |
| `rl_training.py` | implements the crude-RL strategy and its memory-augmented version. |
| `sl_training.py` | implements the semantic loss of a stochastic version (see Prop.2 in the paper). |
| `sup_training.py` | implements the supervised training. |
| `fs_training.py` | implements our stage I training, i.e., softened nesy training with projected MCMC. |
| `bs_training.py` | implements our stage II training, i.e., semi-supervised training with pseudo label (see App.B1 in the paper). |

