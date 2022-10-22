"""Script to run biased lexical overlap model
"""

import argparse
import logging
import pickle
from os import mkdir
from os.path import exists, join
import json

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

from load_word_vectors import load_word_vectors

def is_subseq(needle, haystack):
  l = len(needle)
  if l > len(haystack):
      return False
  else:
      return any(haystack[i:i+l] == needle for i in range(len(haystack)-l + 1))

def load_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))

    for line in data:
        line["document"] = line["document"].split()
        line["question"] = line["question"].split()
        for i in range(1, 5):
            line["option" + str(i)] = line["option" + str(i)].split()

    return data

def build_bias_only(out_dir, train_file, validation_files, w2v_cache=None):
    """Builds bias-only model and saves its predictions
    :param out_dir: Directory to save the predictions
    :param w2v_cache: Cache w2v features to this file
    """
  
    dataset_to_examples = {}
    dataset_to_examples["train"] = load_data(train_file)
    for i, validation_file in enumerate(validation_files.split(",")):
        dataset_to_examples["dev"+str(i)] = load_data(validation_file)
  
    # Load the pre-normalized word vectors to use when building features
    if w2v_cache and exists(w2v_cache):
        with open(w2v_cache, "rb") as f:
            w2v = pickle.load(f)
    else:
        logging.info("Loading word vectors")
        voc = set()
        for k, v in dataset_to_examples.items():
            #if k=="dev3":
            #    continue
            #if "dev" in k:
            #    continue
            for ex in v:
                voc.update(ex["document"])
                voc.update(ex["question"])
                for i in range(1, 5):
                    voc.update(ex["option"+str(i)])

        words, vecs = load_word_vectors("crawl-300d-2M", voc)
        w2v = {w: v/np.linalg.norm(v) for w, v in zip(words, vecs)}
        if w2v_cache:
            with open(w2v_cache, "wb") as f:
                pickle.dump(w2v, f)
  
    # Build the features, store as a pandas dataset
    dataset_to_features = {}
    for name, examples in dataset_to_examples.items():
        print("Building features for %s.." % name)
        features = []
        for example in examples:
            context = []
            for x in example["document"]:
                context.append(x.lower())
            for x in example["question"]:
                context.append(x.lower())
            context_words = set(context)
           
            ops = {}
            for i in range(1, 5):
                ops[i] = [tok.lower() for tok in example["option" + str(i)]]

                n_words_in_context = sum(x in context_words for x in ops[i])
                fe = {
                  "h-is-subseq": is_subseq(ops[i], context),
                  "all-in-p": n_words_in_context == len(ops[i]),
                  "percent-in-p": n_words_in_context / len(ops[i]),
                  "log-len-diff": np.log(max(len(context) - len(ops[i]), 1)),
                  "label": 1 if i==int(example["label"][-1]) else 0
                }
  
                h_vecs = [w2v[w] for w in context if w in w2v]
                p_vecs = [w2v[w] for w in ops[i] if w in w2v]
                if len(h_vecs) > 0 and len(p_vecs) > 0:
                    h_vecs = np.stack(h_vecs, 0)
                    p_vecs = np.stack(p_vecs, 0)
                    # [h_size, p_size]
                    similarities = np.matmul(h_vecs, p_vecs.T)
                    # [h_size]
                    similarities = np.max(similarities, 1)
                    similarities.sort()
                    fe["average-sim"] = similarities.sum() / len(ops[i])
                    fe["min-similarity"] = similarities[0]
                    if len(similarities) > 1:
                        fe["min2-similarity"] = similarities[1]
  
                features.append(fe)
  
        dataset_to_features[name] = pd.DataFrame(features)
        dataset_to_features[name].fillna(0.0, inplace=True)
  
    # Train the model
    print("Fitting...")
    train_df = dataset_to_features["train"]
    feature_cols = [x for x in train_df.columns if x != "label"]
  
    # C=100 means no regularization
    lr = LogisticRegression(multi_class="auto", solver="liblinear",
                            class_weight='balanced', C=100, random_state=1)
    lr.fit(train_df[feature_cols].values, train_df.label.values)
  
    # Save the model predictions
    if not exists(out_dir):
        mkdir(out_dir)
  
    for name, ds in dataset_to_features.items():
        print("Predicting for %s" % name)
        examples = dataset_to_examples[name]
        pred = lr.predict_proba(ds[feature_cols].values).astype(np.float32)
        y = ds.label.values
  
        bias = {}
        for i in range(len(pred)):
            bias[i] = pred[i]
  
        acc = np.mean(y == np.argmax(pred, 1))
        print("%s two-class accuracy: %.4f (size=%d)" % (name, acc, len(examples)))
  
        if "dev" in name:
            dev_preds = pred[:,1]
        #with open(join(out_dir, "%s.pkl" % name), "wb") as f:
        #    pickle.dump(bias, f)
            gold_labels = [int(example["label"][-1]) for example in dataset_to_examples[name]]
            j = 0
            correct = []
            for i in range(0, dev_preds.shape[0], 4):
                pred_op = np.argmax(dev_preds[i:i+4])
                if (pred_op+1 == gold_labels[j]):
                    correct.append(1)
                else:
                    correct.append(0)
                j += 1
 
            #if name == "dev0":
            #    np.save("overlap_models/source_overlap_transf", np.array(correct))
            
            print("%s qa accuracy: %.4f" % (name, sum(correct) / len(gold_labels)))

def main():
    parser = argparse.ArgumentParser("Train our bias-only model")
    parser.add_argument("--output_dir", default="overlap_models/", help="Directory to store the bias-only predictions")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_files", required=True)
    parser.add_argument("--cache_w2v_features")
    args = parser.parse_args()
  
    build_bias_only(args.output_dir, args.train_file, args.validation_files, args.cache_w2v_features)


if __name__ == "__main__":
    main()

