import networkx as nx
import numpy as np
import torch
from datetime import datetime as dt
import os

from experiment_runner import ExperimentRunner
from data_manager import load_dataset
from node import PDGTNode
from graph_creator import create_graph
from plotter import Plotter
import dill as pickle

'''
In this experiment we test the performance of our algorithm on the non-convex problem of matrix factorization
'''

if __name__ == "__main__":
    experiment_name = "{}_{}".format(os.path.basename(__file__), dt.now().strftime("%m-%d-%Y-%H-%M-%S"))
    experiment_directory_name = "experiment_outputs"

    #### Create the graph
    m = 50
    G = create_graph(m)
    max_degree = max([deg_info[1] for deg_info in list(G.degree)])
    A = np.array(nx.adjacency_matrix(G).todense())
    degrees = A.sum(axis=1)
    W = 0.01 / (max_degree + 1) * A + np.eye(m) - 0.01 / (max_degree + 1) * np.diag(degrees)

    #### Load the dataset, and set the desired target rank
    rank = 40
    # Load the MovieLens dataset from kaggle: https://www.kaggle.com/prajitdatta/movielens-100k-dataset
    M = load_dataset('data/M.p')
    l, n = M.shape
    #----Random initialization
    # U = torch.rand((l,r), requires_grad=True)
    # U.data *= 1./torch.norm(U)
    # V = torch.rand((n,r), requires_grad=True)
    # V.data *= 1./torch.norm(V)
    # Initiallizing close to the saddle 0,0
    U = torch.ones((l, rank), requires_grad=True)
    V = torch.ones((n, rank), requires_grad=True)
    U.data /= -10000
    V.data /= 10000

    # Number of iterations for each phase
    T1 = 1500
    T2 = 150

    threshold_to_add_noise = 1e-6
    noise_ball_radius = 40
    # For simplicity of exposition we check every round for a FOSP although theoretical
    # results show that a logarithmic number of checks suffice ro find a FOSP w.h.p.
    num_criterion_checking_rounds = T1
    # num_criterion_checking_rounds = np.log(T1)
    eta_phase_1 = 3
    eta_phase_2 = eta_phase_1
    num_sgd_updates_per_node_per_round = 1
    phase_2_enabled = False
    node_class = PDGTNode

    runner = ExperimentRunner(M=M,
                              rank=rank,
                              G=G,
                              W=W,
                              node_class=node_class,
                              U=U,
                              V=V,
                              T1=T1,
                              threshold_to_add_noise=threshold_to_add_noise,
                              noise_ball_radius=noise_ball_radius,
                              num_criterion_checking_rounds=num_criterion_checking_rounds,
                              T2=T2,
                              eta_phase_1=eta_phase_1,
                              eta_phase_2=eta_phase_2,
                              num_sgd_updates_per_node_per_round=num_sgd_updates_per_node_per_round,
                              phase_2_enabled=phase_2_enabled)
    loss_values, term_1_values, term_2_values, criterion_values, phases = runner.run()

    if not os.path.exists(experiment_directory_name):
        os.makedirs(experiment_directory_name)
    with open("{}/{}.p".format(experiment_directory_name, experiment_name), 'wb') as f:
        pickle.dump({
            "loss": loss_values,
            "t1": term_1_values,
            "t2": term_2_values,
            "criterion": criterion_values,
            "phases": phases
        }, f)

    # To load this file, make a new file, and start with the following:
    # file_to_load = "<YOUR FILE NAME HERE>"
    # with open("{}/{}".format(experiment_directory_name, file_to_load), 'rb') as f:


    plotter = Plotter(plot_title=experiment_name, output_dir=experiment_directory_name)
    plotter.plot(loss_values=loss_values,
                 term_1_values=term_1_values,
                 term_2_values=term_2_values,
                 criterion_values=criterion_values,
                 phases=phases)