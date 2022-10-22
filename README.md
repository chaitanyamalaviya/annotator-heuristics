# Cascading Biases: Investigating the Effect of Heuristic Annotation Strategies on Data and Models

## Data

All relevant data used in our paper can be found under the `data` directory. Annotator IDs are anonymized.
  - Our collected data can be found at data/complete_data.json.
  - The baseline data used to finetune the biased models can be found at data/baseline_all_fixed.json.
  - The CRT results can be found at data/crt_results.csv.
  - The biased human annotations can be found at data/human_biased_annotations.txt.
  - The question annotations in sec. 7 of the paper can be found at data/question_annotations.log.

## Code 

### Models

  - The scripts to run the partial input models used in our paper can be run using the scripts at code/bash_scripts.
  - The trained model outputs for each model (indicating whether the biased model solves the example correctly) are under data/predictions.

### Annotator Splits

  - The script to run the annotator splits experiments is at code/bash_scripts/run_race_annot_split.sh.

### Analysis

  - All other analyses presented in our paper can be found in the jupyter notebook at code/analysis.ipynb.


## Bibtex
```
@article{malaviya2022cascading,
  title={Cascading Shortcuts: Investigating the Effect of Heuristic Annotation Strategies on Data and Models},
  author={Malaviya, Chaitanya and Bhatia, Sudeep and Yatskar, Mark},
  journal={Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2022}
}
```

