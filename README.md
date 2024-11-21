# research-llm

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Research about new ways of training language models. Introducing intermediate generation tokens (notes) as a way of increasing time the model spends on deducing the answer.

## Project Organization

```
├── LICENSE            <- Open-source license
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

## Experiment 1

**Hypothesis 1:** Large language model trained on corpus with notes models the language better.

**Hypothesis 2:** In comparison, language model trained on corpus with POS tag notes achieves better perplexity when it predicts POS tag prior to word generation rather than post word. 

### Dataset, model architecture & training

**TODO (21.11.2024):** Add corpus specyfication - which one, how many GB / tokens, encoding info, which tagger was used

There were two separate datasets. Fitst, smaller, for initial tests - collection Shakespeare writings. Second, larger, for large scale experiment - fineweb.

Transformer with ~100M parameters was chosen as a model architecture.

Each model was trained on a single NVIDIA GeForce RTX 3080, 10018 MiB. A single model training session took up to 20h

All of the files related to that experiment are stored in `pre_post_experiment` files.

### Design
Experiment was design to compare perplexity of four different types of corpus augmentation:

a) pre notes - model was deducing first POS tag and only then the actual word

`['[RB]', 'Bright', 'ly', '[VBG]', 'jumping', '[NNS]', 'jelly', 'beans', '[VBD]', 'became', '[JJ]', 'jubilant', '[.]', '.']`

b) post notes - model was generating word tokens then giving the POS tag that caracterized them

`['Bright', 'ly', '[RB]', 'jumping', '[VBG]', 'jelly', 'beans', '[NNS]', 'became', '[VBD]', 'jubilant', '[JJ]', '.', '[.]']`

c) blank notes - as a reference the model was trained on the corpus with blank notes

`['Bright', 'ly', '[BLANK]', 'jumping', '[BLANK]', 'jelly', 'beans', '[BLANK]', 'became', '[BLANK]', 'jubilant', '[BLANK]', '.', '[BLANK]']`

c) none notes - for the comparison the test were also made with a regular corpus, without any notes; note that this makes the model context window up to two times larger

`['Bright', 'ly', 'jumping', 'jelly', 'beans', 'became', 'jubilant', '.']`


## Experiment 2

**Hypothesis 1:** Amount of information provided in post-notes positively correlates with the model ability to predict natural language.

### Design

- Use Polish language as its syntax and morphology are richer than English


### TODO

- create different tagging procedures for Polish language - make a POC notebook

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
