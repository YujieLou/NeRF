python -m torch.distributed.launch --nproc_per_node 1 --master_port 60660 main_ddp.py configs/fern.txt --local_rank 3


python -m torch.distributed.launch --nproc_per_node 1 --master_port 60002 main_ddp.py configs/A-good.txt --local_rank 2