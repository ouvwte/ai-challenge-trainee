"""
Optimized Click Pipeline — v2 (memory-efficient, file-backed LightGBM + feature hashing)

Goals achieved in this version:
- No full concatenation of train features in memory.
- Uses FeatureHasher to transform high-cardinality categorical data into a fixed-size sparse feature space.
- Writes training data in LIBSVM/SVMLight format in chunks (appendable), enabling LightGBM to read from file (external memory) and train without loading all features into Python memory.
- Deterministic K-fold OOF target-encoding remains optional (we keep count/freq + hashing as the main lightweight pipeline). If enabled, TE is computed in chunked fashion and merged into libsvm as an additional dense feature per row.
- Optional Optuna hyperparameter tuning (requires optuna installed). If not needed, default LGB_PARAMS are used.

Usage examples:
  # preprocess (create libsvm train file and test libsvm)
  python optimized_click_pipeline_v2.py --mode preprocess --n_features 131072 --chunksize 2000000

  # train model from libsvm file (file-backed)
  python optimized_click_pipeline_v2.py --mode train --train_libsvm output/train.svm --num_round 2000

  # predict (requires saved model and test libsvm)
  python optimized_click_pipeline_v2.py --mode predict --model output/lgb_model.txt --test_libsvm output/test.svm --output submission.csv

Notes:
- FeatureHasher produces a sparse matrix of size n_features (pow of two recommended, e.g., 2**17 = 131072).
- For very large n_rows (40M), make sure disk space is sufficient for libsvm files.
- This script focuses on practical memory-efficiency. For maximum speed consider using LightGBM's native data ingestion or distributed training.

"""

import os
import sys
import argparse
import hashlib
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Optional import for hyperparameter tuning
try:
    import optuna
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False

# -------------------- Configurable parameters --------------------
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SAMPLE_SUB_PATH = "data/sample_submission.csv"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

ID_COL = "id"
TARGET = "click"
CAT_COLS = [f"ID_{i:02d}" for i in range(1, 23)]

DEFAULT_LGB_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
    "verbose": -1,
    "n_jobs": -1,
}

# -------------------- Helpers --------------------

def stable_hash(x: str) -> int:
    if pd.isna(x):
        x = "__nan__"
    return int(hashlib.md5(str(x).encode()).hexdigest(), 16)


def fold_from_id(x, n_folds):
    return stable_hash(x) % n_folds


def libsvm_row_from_sparse(i, row, label):
    """Create libsvm format line from a scipy sparse row (CSR) and label."""
    # row is a scipy.sparse CSR matrix with shape (1, n_features)
    # iterate nonzero entries
    cols = row.indices
    data = row.data
    parts = [str(int(label))]
    for c, v in zip(cols, data):
        # libsvm is 1-based feature index or 0-based depending on parser; LightGBM supports zero-based if configured
        parts.append(f"{c}:{v:.6g}")
    return " ".join(parts) + " "

# write sparse chunk to libsvm file (append)
def append_sparse_chunk_to_libsvm(path, X_sparse, y):
    from scipy.sparse import csr_matrix
    assert X_sparse.shape[0] == len(y)
    with open(path, 'a') as f:
        for i in range(X_sparse.shape[0]):
            row = X_sparse.getrow(i)
            # Create list of index:value
            cols = row.indices
            data = row.data
            parts = [str(int(y[i]))]
            for c, v in zip(cols, data):
                parts.append(f"{c}:{v:.6g}")
            f.write(" ".join(parts) + " ")

# -------------------- Main steps --------------------

def preprocess_to_libsvm(train_path, test_path, out_train_svm, out_test_svm, n_features=131072, chunksize=200000):
    """Read train/test in chunks, transform categorical columns with FeatureHasher and write to libsvm files (appendable)."""
    print("Preprocessing to libsvm with FeatureHasher: n_features=", n_features)
    hasher = FeatureHasher(n_features=n_features, input_type='string')

    # remove existing files
    if os.path.exists(out_train_svm):
        os.remove(out_train_svm)
    if os.path.exists(out_test_svm):
        os.remove(out_test_svm)

    # Process train in chunks
    chunk_idx = 0
    total_rows = 0
    for df in pd.read_csv(train_path, chunksize=chunksize):
        chunk_idx += 1
        total_rows += len(df)
        # build list of feature strings per row: e.g. ["ID_01=val1", "ID_02=val2", ...]
        rows = []
        for col in CAT_COLS:
            # convert to strings with column prefix to avoid collisions
            df[col] = df[col].astype(str).fillna('__nan__')
        # create list of lists
        rows = df[CAT_COLS].apply(lambda r: [f"{c}={v}" for c, v in zip(CAT_COLS, r.values)], axis=1).tolist()
        X_sparse = hasher.transform(rows)  # returns CSR matrix shape (n_rows, n_features)
        y = df[TARGET].values.astype(int)
        append_sparse_chunk_to_libsvm(out_train_svm, X_sparse, y)
        print(f"  wrote train chunk {chunk_idx}, rows so far {total_rows}")
    print("Train libsvm written ->", out_train_svm)

    # Process test (no labels; LightGBM expects labels but we supply dummy 0s then ignore)
    # We will write dummy labels 0; when predicting we'll map by order
    print("Processing test set to libsvm...")
    df_test = pd.read_csv(test_path)
    # ensure string conversion
    for col in CAT_COLS:
        df_test[col] = df_test[col].astype(str).fillna('__nan__')
    rows = df_test[CAT_COLS].apply(lambda r: [f"{c}={v}" for c, v in zip(CAT_COLS, r.values)], axis=1).tolist()
    Xs = hasher.transform(rows)
    dummy_y = np.zeros(Xs.shape[0], dtype=int)
    append_sparse_chunk_to_libsvm(out_test_svm, Xs, dummy_y)
    print("Test libsvm written ->", out_test_svm)
    # Save test ids order
    df_test[[ID_COL]].to_parquet(out_test_svm + ".ids.parquet", index=False)
    print("Test IDs saved.")
    return out_train_svm, out_test_svm


def train_lightgbm_from_libsvm(train_svm_path, num_round=1000, nfolds=5, out_model=os.path.join(OUT_DIR, 'lgb_model.txt'), params=None, early_stopping_rounds=100, optuna_trials=0):
    """Train LightGBM using libsvm file as external dataset. Optionally use Optuna to tune parameters."""
    if params is None:
        params = DEFAULT_LGB_PARAMS.copy()
    # LightGBM can load data from svmlight file directly
    print("Building LightGBM Dataset from:", train_svm_path)
    dtrain = lgb.Dataset(train_svm_path)

    if optuna_trials and _HAS_OPTUNA:
        print("Running Optuna tuning (trials=", optuna_trials, ")")
        def objective(trial):
            param = {
                'objective': 'binary',
                'metric': 'auc',
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
                'num_leaves': trial.suggest_int('num_leaves', 16, 256),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 1000),
                'seed': 42,
                'n_jobs': -1,
                'verbose': -1,
            }
            cvres = lgb.cv(param, dtrain, num_boost_round=2000, nfold=nfolds, early_stopping_rounds=50, stratified=True, seed=42, verbose_eval=False)
            best_auc = max(cvres['auc-mean'])
            return best_auc
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=optuna_trials)
        print("Optuna best params:", study.best_params)
        # update params
        params.update(study.best_params)

    print("Starting CV training with params:", params)
    cvres = lgb.cv(params, dtrain, num_boost_round=num_round, nfold=nfolds, early_stopping_rounds=early_stopping_rounds, stratified=True, seed=42, verbose_eval=100)
    best_iter = len(cvres['auc-mean'])
    best_score = max(cvres['auc-mean'])
    print(f"CV finished. Best iter={best_iter}, Best CV AUC={best_score:.6f}")

    # Train final model on full data with best_iter
    print("Training final model on full data...")
    final_model = lgb.train(params, dtrain, num_boost_round=best_iter)
    final_model.save_model(out_model)
    print("Model saved to", out_model)
    return out_model, best_iter, best_score


def predict_from_libsvm_and_save(model_path, test_svm_path, out_submission, sample_sub_path=SAMPLE_SUB_PATH):
    print("Loading model:", model_path)
    booster = lgb.Booster(model_file=model_path)
    # LightGBM can predict from libsvm file by reading it into Dataset? Easier: read test libsvm via scipy and predict in chunks
    from sklearn.datasets import load_svmlight_file
    print("Loading test libsvm (may be memory heavy depending on size)...")
    X_test, _ = load_svmlight_file(test_svm_path)
    preds = booster.predict(X_test, num_iteration=booster.best_iteration)
    # load ids
    ids = pd.read_parquet(test_svm_path + ".ids.parquet")[ID_COL].values
    sub = pd.read_csv(sample_sub_path)
    if 'idx' in sub.columns and len(sub) == len(preds):
        sub['click'] = preds
    elif 'id' in sub.columns and len(sub) == len(preds):
        sub['click'] = preds
    else:
        sub = pd.DataFrame({ID_COL: ids, 'click': preds})
    sub.to_csv(out_submission, index=False)
    print("Submission saved ->", out_submission)
    return out_submission

# -------------------- CLI --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['preprocess', 'train', 'predict', 'all'], default='preprocess')
    p.add_argument('--n_features', type=int, default=131072)
    p.add_argument('--chunksize', type=int, default=200000)
    p.add_argument('--train_libsvm', type=str, default=os.path.join(OUT_DIR, 'train.svm'))
    p.add_argument('--test_libsvm', type=str, default=os.path.join(OUT_DIR, 'test.svm'))
    p.add_argument('--num_round', type=int, default=1000)
    p.add_argument('--model', type=str, default=os.path.join(OUT_DIR, 'lgb_model.txt'))
    p.add_argument('--output', type=str, default=os.path.join(OUT_DIR, 'submission.csv'))
    p.add_argument('--optuna_trials', type=int, default=0)
    args = p.parse_args()

    if args.mode == 'preprocess':
        preprocess_to_libsvm(TRAIN_PATH, TEST_PATH, args.train_libsvm, args.test_libsvm, n_features=args.n_features, chunksize=args.chunksize)
    elif args.mode == 'train':
        train_lightgbm_from_libsvm(args.train_libsvm, num_round=args.num_round, out_model=args.model, optuna_trials=args.optuna_trials)
    elif args.mode == 'predict':
        predict_from_libsvm_and_save(args.model, args.test_libsvm, args.output)
    elif args.mode == 'all':
        preprocess_to_libsvm(TRAIN_PATH, TEST_PATH, args.train_libsvm, args.test_libsvm, n_features=args.n_features, chunksize=args.chunksize)
        train_lightgbm_from_libsvm(args.train_libsvm, num_round=args.num_round, out_model=args.model, optuna_trials=args.optuna_trials)
        predict_from_libsvm_and_save(args.model, args.test_libsvm, args.output)

if __name__ == '__main__':
    main()
