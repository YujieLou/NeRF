python -m torch.distributed.launch --nproc_per_node 1 --master_port 60660 main_ddp.py configs/fern.txt --local_rank 3


python -m torch.distributed.launch --nproc_per_node 1 --master_port 60002 main_ddp.py configs/A-good.txt --local_rank 2

python -m torch.distributed.launch --nproc_per_node 4 --master_port 60660 main_ddp.py configs/ops3.txt --gpus

python -m torch.distributed.launch --nproc_per_node 1 --master_port 60660 main_ddp.py configs/opl3.txt --local_rank 3
python -m torch.distributed.launch --nproc_per_node 1 --master_port 60661 main_ddp.py configs/opl3.txt --local_rank 2
python -m torch.distributed.launch --nproc_per_node 1 --master_port 60662 main_ddp.py configs/opl3.txt --local_rank 1
python -m torch.distributed.launch --nproc_per_node 1 --master_port 60663 main_ddp.py configs/opl3.txt --local_rank 0

