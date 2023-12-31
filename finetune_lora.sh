#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

################## vicuna-v1.5 ##################
# MODEL_VERSION="llama-2-7b-chat"
################## vicuna-v1.5 ##################


# the mm_mlp_adapter is ignored
#--pretrain_mm_mlp_adapter /mnt/private_yucheng/huggingface_hub/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5/mm_projector.bin \

deepspeed --include localhost:4,5 llava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path /root/szd/LLaVA1.5/ckpt/vicuna-7b-1.5-altered \
    --version v1 \
    --data_path /root/szd/LLaVA1.5/box_mlp_altered.json \
    --image_folder /root/szd/LLaVA1.5/playground/data \
    --vision_tower /root/szd/LLaVA1.5/ckpt/openaiclip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /root/szd/LLaVA1.5/ckpt/vicuna-7b-1.5-projector/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --fp16 True \
    --lora_r 64 \
    --output_dir ./ckpt/llava-v1.5-from-pretrained-finetune_lora-1103-1808 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 
