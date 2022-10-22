# Train lexical overlap model on baseline data and evaluate on our data

python run_overlap_model.py \
--train_file data/baseline_all_fixed.json \
--validation_files data/complete_data.json,data/complete_proc.json

