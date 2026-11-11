# Graph-based time-series anomaly detection

Run the three numbered scripts in order from this directory. Supporting modules
must remain alongside them.

## Installation
Recommend to use Python 3.10.

## macOS
For a new environment, start with an Apple Silicon (osx-arm64) Miniconda installation:

```sh
conda create -n venv -c conda-forge python=3.10 numpy=1.26.4 scipy=1.15.2
conda activate venv
python -m pip install -r requirements.txt
```

Confirm that SciPy and the main dependencies load:

```sh
python -c "import scipy; from scipy.sparse.linalg import svds; import tensorflow; import sklearn_extra; print('Imports OK:', scipy.__version__)"
```

Run the numbered scripts with `python` using the commands below. 
Apple GPU acceleration was not validated.

## Windows
The numbered workflow is expected to work on 64-bit Windows with Python 3.10
and CPU execution. From PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```


## Input data
Prepare SCADA data as `data/train.csv` and `data/test.csv` with matching numeric sensor columns.
Step 1 drops common timestamp columns and keeps numeric columns. Exclude labels
Optional `--standardize` uses training statistics for both splits.

Prepare `data/labels.csv` about the label as a failure indicator (0 normal, 1 anomaly).
Keep the same row selection across test data and labels. For step 3, use step 1's
`--step_size 1`; other strides require changes to label alignment.

## 1. Generate graphs (example code)

```sh
python 1_save_cosine_graphs_with_sigmoid.py \
  --train_csv data/train.csv --test_csv data/test.csv \
  --base_dir cosine_graphs --method cosine_row --standardize \
  --ws_start 24 --ws_end 96 --ws_step 1 \
  --thr_start 0.8 --thr_end 0.88 --thr_step 0.005
```
Graphs and windows are saved under `cosine_graphs/w{size}/t{threshold}/{train,test}/`.

## 2. Select the best threshold and window size 

```sh
python 2_select_wds_ths.py \
  --base_dir cosine_graphs \
  --ws_start 24 --ws_end 96 --ws_step 1 \
  --thr_start 0.8 --thr_end 0.88 --thr_step 0.005
```

Use the same grid and directory as step 1. The script chooses the threshold
with the largest GAE-embedded k-medoids loss for each window size and writes
`best_thresholds.csv`, `all_threshold_losses.csv`, and per-window loss tables.
From `best_thresholds.csv`, choose the values of TH and w.


## 3. Score and evaluate

The following is an example only: replace 24 and 0.8 with your selected values from step 2.
The corresponding graph folders must exist. Choose h values no larger than the
number of training graphs

```sh
python 3_run_graphs.py \
  --train_path cosine_graphs/w24/t0.8/train/graphs.pkl \
  --test_path cosine_graphs/w24/t0.8/test/graphs.pkl \
  --labels_csv data/labels.csv --window_size 24 --best_threshold 0.8 \
  --eval_start 24 --h_start 20 --h_end 30 --h_step 1 \
  --n_workers 1 --out_dir results
```

Step 3 uses `score_worker_old.py` to score sequential training collections.
It saves per-h scores and metrics, `summary.csv`, best-result CSVs, plots, and
timing measurements. `score_worker_twoside.py` is an optional helper and is not
called by step 3.

## Reproducibility and scope
Step 2 uses seed 42; step 3 accepts `--seed` (default 42). 

`pa_rpa_metrics.py` uses an external Merlion helper if both Merlion and
`../AD_ref/evaluate_all_metrics.py` are available; otherwise it uses its bundled
NumPy implementation. A standalone checkout uses the bundled implementation.
Do not compare runs across those paths without checking metric equivalence.

`gae/train.py` is a legacy entry point requiring its original input_data module;
the numbered workflow uses `gae/embedding.py` instead.

## Licenses

For gae.py and embedding.py, we use the code from: https://bit.ly/RWiLS_NCD
