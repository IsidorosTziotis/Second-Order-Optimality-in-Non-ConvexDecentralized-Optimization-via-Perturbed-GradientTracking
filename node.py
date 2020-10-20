import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy


class Node:
    def __init__(self, node_id, K, M, rank, U=None, V=None):
        self.node_id = node_id # The id of the node in the graph object
        self.K = K # total number of users in the system
        self.neighbors = [] # list of neighbor nodes
        self.everyone = [] # list of neighbor nodes
        self.M = M.float() # the input data matrix we want to fit U V^T to
        self.l, self.n = self.M.shape
        self.r = rank # the desired rank of the representation

        self.has_updates_from_everyone = False # indicates whether or not y values come from every node, or just from the neighboring nodes
        self.W_everyone = 1./K * np.ones((K,K))

        num_users_before_me = int(self.node_id * np.floor(self.l / self.K))
        num_users_for_me = int(min(np.floor(self.l/self.K), self.l - num_users_before_me))
        num_users_after_me = self.l - num_users_before_me - num_users_for_me
        self.users_for_node = np.concatenate([[0]*num_users_before_me, [1]*num_users_for_me, [0]*num_users_after_me])

        self.loss_fn = nn.MSELoss()

        self.I_users = torch.diag(torch.tensor(self.users_for_node)).float() # Indicator matrix of the users for this node
        self.U = torch.rand((self.l, self.r), requires_grad=True) if U is None else deepcopy(U)
        self.V = torch.rand((self.n, self.r), requires_grad=True) if V is None else deepcopy(V)
        self.U.float()
        self.V.float()

        self.U_proposed_updates = None
        self.V_proposed_updates = None

    def reset_y(self):
        print("Not implemented")
        assert False

    def recompute_gradients(self):
        # Zero out the gradient buffer
        self.U.grad.zero_() if self.U.grad is not None else None
        self.V.grad.zero_() if self.V.grad is not None else None
        # Compute the loss at x^i_{t-1}
        loss = self.calculate_loss()
        # Compute the gradient at the current iterate
        loss.backward()

    def get_degree(self):
        return len(self.neighbors)

    def add_neighbor(self, node: "Node"):
        self.neighbors.append(node)

    def add_everyone(self, custom_nodes: ["Node"]):
        self.everyone = [node for node in custom_nodes.values() if node.node_id != self.node_id]

    def get_model_copy(self):
        return deepcopy(self.U), deepcopy(self.V)

    def average_y(self):
        print("Not implemented")
        assert False

    def gather_model_updates_from_neighbors(self):
        print("Not implemented")
        assert False

    def gather_only_x_updates_from_neighbors(self):
        print("Not implemented")
        assert False

    def gather_model_updates_from_everyone(self):
        print("Not implemented")
        assert False

    def gather_only_y_updates_from_everyone(self):
        print("Not implemented")
        assert False

    def apply_model_updates(self):
        self.U = self.U_proposed_updates
        self.V = self.V_proposed_updates

    def calculate_loss(self):
        return self.K * self.loss_fn(torch.mm(torch.mm(self.I_users, self.U), self.V.t()), torch.mm(self.I_users, self.M))

    def optimize(self, num_steps, step_size):
        print("Not implemented")
        assert False

#The following class is not used in our ain algorithm
class SGDNode(Node):
    def gather_model_updates_from_neighbors(self):
        # Set the buffer variables
        self.U_proposed_updates = deepcopy(self.U)
        self.V_proposed_updates = deepcopy(self.V)
        # Update this node's model by averaging the iterates from the other model
        scale_factor = 1./(len(self.neighbors) + 1)
        self.U_proposed_updates.data *= scale_factor
        self.V_proposed_updates.data *= scale_factor
        for neighbor in self.neighbors:
            neighbor_U, neighbor_V = neighbor.get_model_copy()
            self.U_proposed_updates.data += scale_factor * neighbor_U.data
            self.V_proposed_updates.data += scale_factor * neighbor_V.data

    def optimize(self, num_steps, step_size):
        # Run SGD for this particular node
        for step in range(num_steps):
            # Zero out the gradient buffer
            if self.U.grad is None or self.V.grad is None:
                loss = self.calculate_loss()
                # loss = self.loss_fn(torch.mm(self.U, self.V.t()), self.M)
                loss.backward()
            self.U.grad.zero_() if self.U.grad is not None else None
            self.V.grad.zero_() if self.V.grad is not None else None

            # Compute the loss at the current iterate
            loss = self.calculate_loss()
            # Store the loss for the node
            self.curr_loss = loss.item()
            # Compute the gradient at the current iterate
            loss.backward()
            # SGD parameter update step
            self.U.data -= step_size * self.U.grad.data
            self.V.data -= step_size * self.V.grad.data


class PDGTNode(Node):
    def __init__(self, node_id, K, M, rank, W, U=None, V=None):
        super().__init__(node_id=node_id, K=K, M=M, rank=rank, U=U, V=V)
        self.W = np.array(W)
        self.U_y = None
        self.V_y = None
        self.prev_grad_U = None
        self.prev_grad_V = None
        self.reset_y()

        self.U_neighbors = {}
        self.V_neighbors = {}
        self.U_y_neighbors = {}
        self.V_y_neighbors = {}

    def reset_y(self):
        self.recompute_gradients()
        self.U_y = deepcopy(self.U.grad)
        self.V_y = deepcopy(self.V.grad)

    def average_y(self):
        weights_to_use = self.W_everyone if self.has_updates_from_everyone else self.W
        self.U_y.data *= weights_to_use[self.node_id, self.node_id]
        for neighbor_id in self.U_y_neighbors.keys():
            self.U_y.data += weights_to_use[self.node_id, neighbor_id] * self.U_y_neighbors[neighbor_id].data
        self.V_y.data *= weights_to_use[self.node_id, self.node_id]
        for neighbor_id in self.V_y_neighbors.keys():
            self.V_y.data += weights_to_use[self.node_id, neighbor_id] * self.V_y_neighbors[neighbor_id].data

    def get_model_copy(self):
        return deepcopy(self.U), deepcopy(self.V), deepcopy(self.U_y), deepcopy(self.V_y)

    def _gather_model_updates(self, nodes: ["Node"], update_x=True, update_y=True):
        if update_x:
            self.U_neighbors = {}
            for neighbor in nodes:
                neighbor_U, neighbor_V, _, _ = neighbor.get_model_copy()
                if neighbor.node_id not in self.U_neighbors:
                    # Initialize the map to have an entry for neighbor.node_id
                    self.U_neighbors[neighbor.node_id] = None
                    self.V_neighbors[neighbor.node_id] = None
                self.U_neighbors[neighbor.node_id] = deepcopy(neighbor_U)
                self.V_neighbors[neighbor.node_id] = deepcopy(neighbor_V)
        if update_y:
            self.U_y_neighbors = {}
            for neighbor in nodes:
                _, _, neighbor_U_y, neighbor_V_y = neighbor.get_model_copy()
                if neighbor.node_id not in self.U_y_neighbors:
                    # Initialize the map to have an entry for neighbor.node_id
                    self.U_y_neighbors[neighbor.node_id] = None
                    self.V_y_neighbors[neighbor.node_id] = None
                self.U_y_neighbors[neighbor.node_id] = deepcopy(neighbor_U_y)
                self.V_y_neighbors[neighbor.node_id] = deepcopy(neighbor_V_y)

    def gather_only_x_updates_from_neighbors(self):
        self._gather_model_updates(nodes=self.neighbors, update_x=True, update_y=False)

    def gather_model_updates_from_neighbors(self):
        self._gather_model_updates(nodes=self.neighbors, update_x=True, update_y=True)
        self.has_updates_from_everyone = False

    def gather_model_updates_from_everyone(self):
        self._gather_model_updates(nodes=self.everyone, update_x=True, update_y=True)
        self.has_updates_from_everyone = True

    def gather_only_y_updates_from_everyone(self):
        self._gather_model_updates(nodes=self.everyone, update_x=False, update_y=True)
        self.has_updates_from_everyone = True

    def apply_model_updates(self):
        # Do nothing
        return True

    def inject_noise(self, noise_U, noise_V):
        self.U.data += noise_U.data
        self.V.data += noise_V.data

    def optimize(self, num_steps, step_size):
        # Run SGD for this particular node
        for step in range(num_steps):
            self.recompute_gradients()

            # Keep track of grad f_i(x^i_{t-1})
            self.prev_grad_U = deepcopy(self.U.grad)
            self.prev_grad_V = deepcopy(self.V.grad)

            # Compute x^i_t for U
            self.U.data *= self.W[self.node_id,self.node_id]
            for neighbor_id in self.U_neighbors.keys():
                self.U.data += self.W[self.node_id,neighbor_id] * self.U_neighbors[neighbor_id].data
            self.U.data -= step_size * self.U_y.data
            # Compute x^i_t for V
            self.V.data *= self.W[self.node_id,self.node_id]
            for neighbor_id in self.V_neighbors.keys():
                self.V.data += self.W[self.node_id,neighbor_id] * self.V_neighbors[neighbor_id].data
            self.V.data -= step_size * self.V_y.data

            self.recompute_gradients()

            # Compute y^i_t
            self.average_y()
            self.U_y.data += self.U.grad.data - self.prev_grad_U
            self.V_y.data += self.V.grad.data - self.prev_grad_V

