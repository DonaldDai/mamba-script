#!/bin/bash
#SBATCH --job-name=sc_mamba_finetune
#SBATCH --nodes=1
#SBATCH -p gpu-a100 
#SBATCH -t 48:00:00 
#SBATCH --mail-user=daizhilian@hotmail.com
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --output=./slurmlogs/%x_%j_nodes_%N.out 
#SBATCH --error=./slurmlogs/%x_%j_nodes_%N.err 


# 初始化 Conda 环境
source /work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh

# 激活你的 Conda 环境
conda activate mamba

# 切换到特定的工作目录
cd /work/09735/yichao/ls6/zhilian/mamba-script

# 获取主节点的地址
MASTER_ADDR=$(hostname)

# 执行PyTorch分布式训练命令
srun bash -c "torchrun --nproc_per_node=3 --nnodes=1 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 finetune_zl.py"


