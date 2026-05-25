# RCDNet

This repository contains the training, evaluation, data-loading, model, loss, learning-rate scheduling, and complexity-checking code for RCDNet.

## Project structure

```text
RCDNet\_release/
├── main.py
├── data\_loading.py
├── requirements.txt
├── scripts/
│   ├── \_\_init\_\_.py
│   ├── train.py
│   ├── test.py
│   └── complexity\_check.py
└── model/
    ├── \_\_init\_\_.py
    ├── arch.py
    ├── losses.py
    └── scheduler.py
```

## Requirements

Install the core dependencies with:

```bash
pip install -r requirements.txt
```

The code was organized for TensorFlow/Keras training and evaluation.

## Dataset layout

The scripts expect the following relative dataset structure under `./data`:

```text
data/
├── LOLv1/
│   ├── Train/input/\*.png
│   ├── Train/target/\*.png
│   ├── Test/input/\*.png
│   └── Test/target/\*.png
└── LOLv2/
    ├── Real\_captured/
    │   ├── Train/Low/\*.png
    │   ├── Train/Normal/\*.png
    │   ├── Test/Low/\*.png
    │   └── Test/Normal/\*.png
    └── Synthetic/
        ├── Train/Low/\*.png
        ├── Train/Normal/\*.png
        ├── Test/Low/\*.png
        └── Test/Normal/\*.png
```

Supported dataset names are:

* `LOLv1`
* `LOLv2\_Real`
* `LOLv2\_Synthetic`

