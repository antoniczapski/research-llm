# research-llm

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Research about new ways of training language models. Introducing intermediate generation tokens (notes) as a way of increasing compute per answer.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for research-llm-module
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── research-llm-module                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes research-llm-module a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

--------

## Running on remote

### Start a new screen session
screen -S training_cuda_0

### Detach from the screen session (Press Ctrl + A, then D)
Ctrl + A, D

### Reconnect to an existing screen session
screen -r training_cuda_0

### List all active screen sessions
screen -ls

### Reconnect to a screen session using its ID
screen -r session_id

### Close a screen session (when inside the session)
exit

### GPU usage monitoring
watch -n 1 nvidia-smi


------------------------------------------------------------------------------

/pio/scratch/1/i317214/miniconda/envs/research-llm-env/bin/python /pio/scratch/1/i317214/research-llm/research-llm-module/modeling/train_none.py

screen -S training_cuda_0_13_11
screen -S training_cuda_1_13_11


screen -r training_cuda_0_13_11
screen -r training_cuda_1_13_11