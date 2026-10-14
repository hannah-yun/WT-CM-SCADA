# Graph-based time-series anomaly detection

Run the three numbered scripts in order from this directory. Supporting modules
must remain alongside them.

## Installation

Use Python 3.10 (the version used by the local graph_detection environment).
The requirements pin direct dependencies from that environment; installation
on a fresh machine still needs verification.

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Apple Silicon / macOS
The three-stage synthetic workflow was validated on an Apple Silicon Mac using
Python 3.10 and a Conda build of SciPy 1.15.2. For a new environment, start with
an Apple Silicon (osx-arm64) Miniconda or Miniforge installation:

```sh
conda create -n graph_detection -c conda-forge python=3.10 numpy=1.26.4 scipy=1.15.2
conda activate graph_detection
python -m pip install -r requirements.txt
```

Confirm that SciPy and the main dependencies load:

```sh
python -c "import scipy; from scipy.sparse.linalg import svds; import tensorflow; import sklearn_extra; print('Imports OK:', scipy.__version__)"
```

Run the numbered scripts with `python` using the commands below. Start with the
synthetic smoke run before processing your own data. These setup commands have
not been tested in a newly created environment; validation used the existing
local environment. Apple GPU acceleration was not validated.

The pip SciPy build previously failed to load `_spropack` on the tested Mac.
Installing SciPy 1.15.2 through Conda resolved that import error. The requirements
file uses the same version, so pip should retain the already-installed Conda
package when its version satisfies the requirement.

## Windows
The numbered workflow is expected to work on 64-bit Windows with Python 3.10
and CPU execution. From PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run each script with `.\.venv\Scripts\python.exe` instead of `python`.
Use one-line commands; the backslash continuations in the examples below are
for Unix shells. Begin with `--n_workers 1` for step 3.

TensorFlow 2.20 supports CPU execution on native Windows. For NVIDIA GPU use,
follow the [TensorFlow WSL2 installation guide](https://www.tensorflow.org/install/pip).
Native Windows GPU support ended after TensorFlow 2.10.

## Input data
Prepare `data/train.csv` and `data/test.csv` with matching numeric sensor columns.
Step 1 drops common timestamp columns and keeps numeric columns. Exclude labels
and other numeric metadata with `--drop_cols`; do not use them as sensor inputs.
Optional `--standardize` uses training statistics for both splits.

Prepare `data/labels.csv` aligned row-for-row with the test CSV, with
`Timestamp_clean` (parseable timestamps) and `health` (0 normal, 1 anomaly).
Keep the same row selection across test data and labels. For step 3, use step 1's
`--step_size 1`; other strides require changes to label alignment.

## 1. Generate graphs

```sh
python 1_save_cosine_graphs_with_sigmoid.py \
  --train_csv data/train.csv --test_csv data/test.csv \
  --base_dir cosine_graphs --method cosine_row --standardize \
  --ws_start 24 --ws_end 96 --ws_step 1 \
  --thr_start 0.8 --thr_end 0.88 --thr_step 0.005
```

Cosine similarities are mapped through sigmoid(2 * S). Graphs and windows are
saved under `cosine_graphs/w{size}/t{threshold}/{train,test}/`.

## 2. Select thresholds

```sh
python 2_select_wds_ths.py \
  --base_dir cosine_graphs \
  --ws_start 24 --ws_end 96 --ws_step 1 \
  --thr_start 0.8 --thr_end 0.88 --thr_step 0.005
```

Use the same grid and directory as step 1. The script chooses the threshold
with the largest GAE-embedded k-medoids loss for each window size and writes
`best_thresholds.csv`, `all_threshold_losses.csv`, and per-window loss tables.
It does not choose a single final window size.

Choose the window size using a predefined rule or separate validation data.
Use that row's threshold in step 3. If comparing window sizes with labeled data,
use a validation split and reserve the test split for the final evaluation.

## 3. Score and evaluate

The following is an example only: replace 24 and 0.8 with your selected values from step 2.
The corresponding graph folders must exist. Choose h values no larger than the
number of training graphs and an evaluation start index at least the window size.

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
called by step 3. Its two-sided thresholds maximize F1 using evaluation labels.
Step 3 also selects its score cutoff using evaluation labels, so these metrics
represent retrospective threshold selection rather than a frozen deployment cutoff.
The graph threshold from step 2 and the anomaly score cutoff are different parameters.

## Reproducibility and scope

Step 2 uses seed 42; step 3 accepts `--seed` (default 42). TensorFlow execution
and platform differences can still affect results. Record commands, data splits,
dependency versions, and hardware with each experiment.

`pa_rpa_metrics.py` uses an external Merlion helper if both Merlion and
`../AD_ref/evaluate_all_metrics.py` are available; otherwise it uses its bundled
NumPy implementation. A standalone checkout uses the bundled implementation.
Do not compare runs across those paths without checking metric equivalence.

`gae/train.py` is a legacy entry point requiring its original input_data module;
the numbered workflow uses `gae/embedding.py` instead.

Keep real datasets and generated outputs out of Git. `examples/` contains 60-row
synthetic CSVs for smoke testing, not for assessing detection performance.


## Validation status

On 12 October 2026, all three numbered scripts completed a synthetic smoke run
in the local Python 3.10 graph_detection environment with SciPy 1.15.2 installed
from Conda. The run used the bundled 60-row example CSVs, window size 6,
graph threshold 0.6, two training epochs, hidden dimensions 4 and 2, collection
size h=2, and one scoring worker. The selected-threshold, score, and summary
CSVs were nonempty and contained finite numeric values; output plots were
created. All bundled Python files passed syntax checks.

## Licenses

See `THIRD_PARTY_NOTICES.md` and the bundled package licenses. See `COPYRIGHT.md` for ownership of the original numbered scripts. 
