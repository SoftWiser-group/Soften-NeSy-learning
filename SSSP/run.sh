# CUDA_VISIBLE_DEVICES=3 python3 sup_training.py --seed 0 --exp_name sup 

# CUDA_VISIBLE_DEVICES=3 python3 rl_training.py --seed 0 --exp_name rl --mode 'MAPO'
# CUDA_VISIBLE_DEVICES=5 python3 sl_training.py --seed 0 --exp_name sl --T 1.0

# CUDA_VISIBLE_DEVICES=5 python3 fs_training.py --seed 0 --exp_name fs --cooling_strategy 'log' --T 1.0
# CUDA_VISIBLE_DEVICES=5 python3 fs_training.py --seed 0 --exp_name fs --cooling_strategy 'exp' --T 1.0
# CUDA_VISIBLE_DEVICES=5 python3 fs_training.py --seed 0 --exp_name fs --cooling_strategy 'linear' --T 1.0

# CUDA_VISIBLE_DEVICES=1 python3 bs_training.py --seed 0 --exp_name bs --net 'checkpoint/mlp_0_sl_log_0.t7'
CUDA_VISIBLE_DEVICES=1 python3 bs_training.py --seed 0 --exp_name bs --net 'checkpoint/mlp_0_fs_linear_0.t7'
CUDA_VISIBLE_DEVICES=1 python3 bs_training.py --seed 0 --exp_name bs --net 'checkpoint/vae_net_0_sup_0.t7'