# Experiment code for CHORUS

This repository contains the code for the experiments in the paper "*CHORUS:
Foundation Models for Unified Data Discovery and Exploration*" to appear in VLDB 2024.

The folders are structured as follows:

- `src`: source code for the core CHORUS routines
- `experiments`: code for measuring quantitative results reported in the paper
- `datasets`: benchmarks files used in this study
- `results`: outputs of experiments as they appear in the paper
- `scripts`: code for experiments involving third-party tools
- `artifacts`: outputs of the experiments. You can download this folder at [this link](https://www.dropbox.com/scl/fi/555v71lverufsit6tidbq/chorus-artifacts.tar.xz?rlkey=jq423g2jbam88wtd5ta5glmwb&dl=0).

You can install the required packages with `pip install -r requirements.txt`.

For example, to reproduce the results of task 3 in the paper (join column prediction), run the following command:

`python experiments/T3_join_prediction/join-column-results-full.py`

