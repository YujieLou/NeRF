# 
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 20031 main_ddp.py configs/opl3.txt \
# --expname 'tt_test' --no_ndc --lrate 0.0001 --epochs 1 --lrate_x 0.5 --test_all --factor 4

#python -m torch.distributed.launch --nproc_per_node 3 --master_port 20031 main_ddp.py configs/opl3.txt \
#--expname 'tt1' --no_ndc --lrate 0.001 --epochs 1 --lrate_x 0.5 --test_all

# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20031 main_ddp.py configs/opl3.txt \
# --expname 'tt4' --no_ndc --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.1 # --test_all 

# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20032 main_ddp.py configs/opl3.txt \
# --expname 'tt5' --no_ndc --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.1 \
# --n_levels 32 --max_resolution 20480

# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20033 main_ddp.py configs/opl3.txt \
# --expname 'tt6' --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.1 # --test_all 

# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20034 main_ddp.py configs/opl3.txt \
# --expname 'tt7' --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.1 --Model 'NeRF'

# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20032 main_ddp.py configs/opl3.txt \
# --expname 'tt8' --no_ndc --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.1 \
# --n_levels 32 --max_resolution 20480 --n_features_per_level 4 --base_resolution 32

# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20032 main_ddp.py configs/opl3.txt \
# --expname 'tt9' --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.1 \
# --n_levels 32 --max_resolution 20480 --n_features_per_level 4 --base_resolution 32

# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20032 main_ddp.py configs/opl3.txt \
# --expname 'tt10' --no_ndc --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.1 \
# --n_levels 32 --max_resolution 20480 --n_features_per_level 4 --base_resolution 32 --log2_hashmap_size 21

# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20032 main_ddp.py configs/opl3.txt \
# --expname 'tt11' --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.1 \
# --n_levels 32 --max_resolution 20480 --n_features_per_level 4 --base_resolution 32 --log2_hashmap_size 21

# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20032 main_ddp.py configs/opl3.txt \
# --expname 'tt11' --no_ndc --contract --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.1 \
# --n_levels 32 --max_resolution 20480 --n_features_per_level 4 --base_resolution 32 --log2_hashmap_size 21


# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20033 main_ddp.py configs/opl3.txt \
# --expname 'tt12' --no_ndc  --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.1 \
# --n_levels 32 --max_resolution 20480 --n_features_per_level 4 --base_resolution 16 --log2_hashmap_size 21


# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20034 main_ddp.py configs/opl3.txt \
# --expname 'tt12' --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.1 \
# --n_levels 32 --max_resolution 20480 --n_features_per_level 4 --base_resolution 16 --log2_hashmap_size 21

# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20035 main_ddp.py configs/opl3.txt \
# --expname 'tt12' --no_ndc --contract --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.1 \
# --n_levels 32 --max_resolution 20480 --n_features_per_level 4 --base_resolution 16 --log2_hashmap_size 21

# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20034 main_ddp.py configs/opl3.txt \
# --expname 'tt7' --no_ndc --contract --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.1 --Model 'NeRF'

# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20034 main_ddp.py configs/opl3.txt \
# --expname 'tt7' --no_ndc --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.1 --Model 'NeRF'

python -m torch.distributed.launch --nproc_per_node 3 --master_port 20036 main_ddp.py configs/opl3.txt \
--expname 'tt13' --no_ndc  --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.2 --n_levels 32 --max_resolution 20480 --n_features_per_level 8 --base_resolution 32 --log2_hashmap_size 21

# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20037 main_ddp.py configs/opl3.txt \
# --expname 'tt13' --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.2 \ 
# --n_levels 32 --max_resolution 20480 --n_features_per_level 8 --base_resolution 32 --log2_hashmap_size 21

# python -m torch.distributed.launch --nproc_per_node 3 --master_port 20038 main_ddp.py configs/opl3.txt \
# --expname 'tt13' --no_ndc --contract --lrate 0.001 --epochs 2 --lrate_x 0.5 --batch_size 4096 --raw_noise_std 0.2 \
# --n_levels 32 --max_resolution 20480 --n_features_per_level 8 --base_resolution 32 --log2_hashmap_size 21