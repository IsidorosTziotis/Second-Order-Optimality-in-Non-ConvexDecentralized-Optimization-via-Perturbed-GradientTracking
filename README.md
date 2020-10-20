# Distributed Nonconvex Optimization

Welcome to the code repo for Distributed Nonconvex Optimization.
This is the code used to generate all experiments in the paper "Second-Order Optimality in Non-Convex Decentralized Optimization via Perturbed GradientTracking". 

## Setup

To begin, download the MovieLens 100K Dataset from [here](https://www.kaggle.com/prajitdatta/movielens-100k-dataset).
The dataset is named `u.data`. Place this dataset in the root of this project. 
Make sure you have installed all packages in `requirements.txt`.
Then, run `data_manager.py` to prepare the dataset. This will create a pickled dataset
in an automatically created folder `data/`. Once you've run this command, you can then run the
experiment in `vanilla_experiment.py`.
