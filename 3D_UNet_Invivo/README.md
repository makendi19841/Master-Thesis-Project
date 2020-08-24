# 3D U-Net in TensorFlow

*Author: Amos Makendi*


## Get Started

### Setup project

- Create new Python3 `virtualenv` (assumes you have `virtualenv` and
`virtualenvwrapper` installed and set up)
- Install dependencies.
- activate virtual environment 'unet3d': conda activate unet3d




### Dataset

- As per the instructions, the training (51) subjects were pooled to form the train dataset (1816 scans in total).
- The evaluation dataset consists of 16 patients with 332 scans.
- The test dataset consists of 05 patients with 82 scans.
- The data were in data/raw/train, data/raw/eval and data/raw/test.

### Explore and preprocess data

- In this notebook, we load in the MRI scans and their segmentations,
build a Dataset object for the train evaluation and test set.
- Then we check some basic stats of the datasets and visualise a few
scans.
- Finally, we carry out our preprocessing steps and save the train, evaluation and
test datasets.


```bash
jupyter notebook "notebooks/data_exploration.ipynb"
```

## Build model

### Train and evaluate

Train model on train set and evaluate it on evaluation set using the base
model architecture (with batchnormalization)


```bash
python src/main.py -model_dir models/base_model -mode train_eval
```

(Without batch normalisation)

```bash
python src/main.py -model_dir models/base_model_no_bn -mode train_eval
```



### Predict

Predict all 05 patients in test set with the trained base model and
save their predictions to model directory.


```bash
python src/main.py -model_dir models/base_model -mode predict - pred_ix 0 1 2 3 4 

```

### Explore predictions

- Open the Jupyter notebook to have a look at test cases

```bash
jupyter notebook "notebooks/model_exploration.ipynb"
```
