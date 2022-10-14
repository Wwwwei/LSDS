# LSDS

This repository contains the Pytorch implementation for the paper **Learning to Search Original subsequences Benefits Shapelets-based Time Series Classification**.
The official version will be released upon publication of the paper.

## Requirements

The recommended requirements for LSDS are specified as follows:
* Python 3.8
* torch==1.8.1
* scipy==1.6.1
* numpy==1.19.2
* pandas==1.0.1
* scikit_learn==0.24.2
* tslearn==0.24.2
* sktime==0.13.0
* torchmetrics==0.9.1

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Data

The datasets can be obtained and put into `datasets/` folder in the following way:

* [128 UCR datasets](http://www.timeseriesclassification.com) should be put into `datasets/` so that each data file can be located by `datasets/UCR/<dataset_name>/<dataset_name>_*.arff`.


## Usage

To train and evaluate LSDS on a dataset, run the following command:

```train & evaluate
python main.py
```