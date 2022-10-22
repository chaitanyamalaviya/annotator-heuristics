#!/bin/bash

export TRANSFORMERS_CACHE=#
export HF_DATASETS_CACHE=#

heuristic=$1
seeds=(1 2 3)

CUDA_VISIBLE_DEVICES=0 python run_qa.py \
--model_name_or_path LIAMF-USP/roberta-large-finetuned-race \
--train_file annotator_splits_new/${heuristic}_train.json \
--validation_file annotator_splits_new/${heuristic}_dev.json \
--do_train \
--do_eval \
--learning_rate 1e-5 \
--num_train_epochs 4 \
--output_dir qa_models/annotator_split_models_new/${heuristic}_model \
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=1 \
--save_total_limit 1 \
--overwrite_output

for seed in "${seeds[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python run_qa.py \
    --model_name_or_path LIAMF-USP/roberta-large-finetuned-race \
    --train_file annotator_splits_new/${heuristic}_rand_train_${seed}.json \
    --validation_file annotator_splits_new/${heuristic}_rand_dev_${seed}.json \
    --do_train \
    --do_eval \
    --learning_rate 1e-5 \
    --num_train_epochs 4 \
    --output_dir qa_models/annotator_split_models_new/${heuristic}_rand_${seed}_model \
    --per_gpu_eval_batch_size=16 \
    --per_device_train_batch_size=1 \
    --save_total_limit 1 \
    --overwrite_output

    CUDA_VISIBLE_DEVICES=0 python run_qa.py \
    --model_name_or_path LIAMF-USP/roberta-large-finetuned-race \
    --train_file annotator_splits_new/${heuristic}_rand_pooled_train_${seed}.json \
    --validation_file annotator_splits_new/${heuristic}_rand_pooled_dev_${seed}.json \
    --do_train \
    --do_eval \
    --learning_rate 1e-5 \
    --num_train_epochs 4 \
    --output_dir qa_models/annotator_split_models_new/${heuristic}_rand_pooled_${seed}_model \
    --per_gpu_eval_batch_size=16 \
    --per_device_train_batch_size=1 \
    --save_total_limit 1 \
    --overwrite_output
done
