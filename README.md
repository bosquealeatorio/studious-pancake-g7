# Studious Pancake Project

Dummy project to review some good coding practices with G7

The project does a basic grid search process registering the experiment params and metrics to MlFlow

# Basic Usage

## Configure environment

Assuming you have conda installed, to create and activate the environment
```
conda env create -f environment.yml
conda activate pancake-env`
```

## Training process

To run the grid search and log the experiments use

```
python src/training.py
```

## MLFlow

To visualize the experiments and register the model

```
mlflow ui
```

After registering the model on the ui, to serve you can use

```
mlflow models serve -m my_model
```

# Contact

Please don't contact me