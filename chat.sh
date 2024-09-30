CUDA_VISIBLE_DEVICES=0 python chat.py \
    --model_name_or_path /model_name_or_path/ \
    --template chatglm2 \
    --finetuning_type full \
    --checkpoint_dir /checkpoint_dir/ \
    --max_source_length 2048 \
    --max_target_length 512 

# --model_name_or_path and --checkpoint_dir 替换到自己的路径
