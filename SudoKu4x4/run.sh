# CUDA_VISIBLE_DEVICES=0 python3 rl_training.py --seed 0 --exp_name rl --mode 'RL'
# CUDA_VISIBLE_DEVICES=0 python3 rl_training.py --seed 0 --exp_name rl --mode 'MAPO'
# CUDA_VISIBLE_DEVICES=0 python3 sl_training.py --seed 0 --exp_name sl --T 1.0
# CUDA_VISIBLE_DEVICES=0 python3 mcmc_training.py --seed 0 --exp_name mcmc --cooling_strategy 'exp' --T 0.1

# CUDA_VISIBLE_DEVICES=0 python3 fs_training.py --seed 0 --exp_name fs --cooling_strategy 'log' --T 0.1
# CUDA_VISIBLE_DEVICES=0 python3 fs_training.py --seed 0 --exp_name fs --cooling_strategy 'exp' --T 0.1
# CUDA_VISIBLE_DEVICES=0 python3 fs_training.py --seed 0 --exp_name fs --cooling_strategy 'linear' --T 0.1

# CUDA_VISIBLE_DEVICES=0 python3 bs_training.py --seed 0 --exp_name bs_tmp --net 'checkpoint/mlp_0_mcmc_exp_best_0.t7'
# CUDA_VISIBLE_DEVICES=0 python3 bs_training.py --seed 0 --exp_name bs_log --net 'checkpoint/mlp_0_fs_log_best_0.t7'
CUDA_VISIBLE_DEVICES=0 python3 bs_training.py --seed 0 --exp_name bs_exp --net 'checkpoint/final_results/mlp_0_fs_exp_best_0.t7'
# CUDA_VISIBLE_DEVICES=0 python3 bs_training.py --seed 0 --exp_name bs_linear --net 'checkpoint/mlp_0_fs_linear_best_0.t7'