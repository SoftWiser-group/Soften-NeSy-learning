# CUDA_VISIBLE_DEVICES=3 python3 rl_training.py --seed 0 --exp_name rl --mode 'RL'

# CUDA_VISIBLE_DEVICES=3 python3 rl_training.py --seed 0 --exp_name rl --mode 'MAPO'

# CUDA_VISIBLE_DEVICES=3 python3 ngs_training.py --seed 0 --exp_name ngs --nstep 6

CUDA_VISIBLE_DEVICES=3 python3 sl_training.py --seed 0 --exp_name sl --T 1.0

CUDA_VISIBLE_DEVICES=4 python3 sl_training.py --seed 0 --exp_name mi --T 1e-6

# CUDA_VISIBLE_DEVICES=3 python3 fs_training.py --seed 0 --exp_name fs --cooling_strategy 'log' --T 1.0

# CUDA_VISIBLE_DEVICES=3 python3 fs_training.py --seed 0 --exp_name fs --cooling_strategy 'exp' --T 1.0

# CUDA_VISIBLE_DEVICES=5 python3 fs_training.py --seed 0 --exp_name fs --cooling_strategy 'linear' --T 1.0


##========================================================================================

# CUDA_VISIBLE_DEVICES=4 python3 bs_training.py --seed 0 --exp_name bs_ngs --net 'checkpoint/sym_net_0_ngs_6-BS_0.t7'

# CUDA_VISIBLE_DEVICES=4 python3 bs_training.py --seed 0 --exp_name bs_log --net 'checkpoint/lenet_0_mi_0.t7'

# CUDA_VISIBLE_DEVICES=4 python3 bs_training.py --seed 0 --exp_name bs_log --net 'checkpoint/lenet_0_sl_0.t7'

# CUDA_VISIBLE_DEVICES=4 python3 bs_training.py --seed 0 --exp_name bs_log --net 'checkpoint/lenet_0_fs_log_best_0.t7'

# CUDA_VISIBLE_DEVICES=4 python3 bs_training.py --seed 0 --exp_name bs_exp --net 'checkpoint/lenet_0_fs_exp_best_0.t7' --data_shuffle True

# CUDA_VISIBLE_DEVICES=4 python3 bs_training.py --seed 0 --exp_name bs_linear --net 'checkpoint/lenet_0_fs_linear_best_0.t7'  --data_shuffle True



