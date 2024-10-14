# Define the set of learning rates and normalization types you want to search through
norm_type=$1
learning_rates=$2
# Set the CUDA devices and other general parameters
export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=0
export NORM_TYPE=$norm_type

# Function to run a single training task

echo "Training with learning rate: $learning_rates, norm type: $norm_type on GPU $gpu"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=29500 torchrun_main.py \
    --model_config configs/llama_1b.json \
    --lr $learning_rates \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 50000 \
    --warmup_steps 5000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adam \
    --grad_clipping 0.0 \
    --run_name "1b_res_${norm_type}_lr${learning_rates}" \
    --save_dir "1b_res_${norm_type}_lr${learning_rates}_1_2"