
# python3 us_training.py --seed 0 --exp_name us
# CUDA_VISIBLE_DEVICES=3 python3 fs_training.py --seed 0 --exp_name fs --cooling_strategy 'exp' --T 10.0

CUDA_VISIBLE_DEVICES=0 python3 bs_training.py --seed 1 --exp_name bs_exp --net 'checkpoint/lenet_0_fs_exp_best_0.t7' 
# CUDA_VISIBLE_DEVICES=0 python3 bs_training.py --seed 2 --exp_name bs_exp --net 'checkpoint/lenet_0_fs_exp_best_0.t7' --dropout True

# CUDA_VISIBLE_DEVICES=0 python3 sup_training.py --seed 0 --e/xp_name sup
# CUDA_VISIBLE_DEVICES=0 python3 sup_training.py --seed 0 --exp_name sup --dropout True
