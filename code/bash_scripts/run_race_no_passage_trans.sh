export TRANSFORMERS_CACHE=##
export HF_DATASETS_CACHE=##

# Train no passage model on baseline data
CUDA_VISIBLE_DEVICES=0 python run_qa.py \
--model_name_or_path LIAMF-USP/roberta-large-finetuned-race \
--train_file data/baseline_all_fixed.json \
--validation_file data/baseline_all_fixed.json \
--do_train \
--do_eval \
--no_passage \
--learning_rate 1e-5 \
--num_train_epochs 4 \
--output_dir qa_models/baseline_no_passage_transf/ \
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=1 \
--save_total_limit 1 \
--overwrite_output

# Evaluate no passage model on control data
CUDA_VISIBLE_DEVICES=0 python run_qa.py \
--model_name_or_path qa_models/baseline_no_passage_transf/ \
--train_file data/complete_data.json \
--validation_file data/complete_data.json \
--do_eval \
--no_passage \
--learning_rate 1e-5 \
--num_train_epochs 4 \
--output_dir qa_models/control_no_passage_transf/ \
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=1 \
--save_total_limit 1 \
--overwrite_output

# Evaluate no passage model on sources data
CUDA_VISIBLE_DEVICES=0 python run_qa.py \
--model_name_or_path qa_models/baseline_no_passage_transf/ \
--train_file data/complete_proc.json \
--validation_file data/complete_proc.json \
--do_eval \
--no_passage \
--learning_rate 1e-5 \
--num_train_epochs 4 \
--output_dir qa_models/sources_no_passage_transf/ \
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=1 \
--save_total_limit 1 \
--overwrite_output

