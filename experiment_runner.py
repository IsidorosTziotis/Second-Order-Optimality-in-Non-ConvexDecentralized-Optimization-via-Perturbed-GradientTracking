import numpy as np
import torch


class ExperimentRunner:
    def __init__(self,
                 M,
                 rank,
                 G,
                 W,
                 node_class,
                 U,
                 V,
                 T1,
                 threshold_to_add_noise,
                 noise_ball_radius,
                 num_criterion_checking_rounds,
                 T2,
                 eta_phase_1,
                 eta_phase_2,
                 num_sgd_updates_per_node_per_round,
                 phase_2_enabled):
        self.M = M
        self.rank = rank # the target rank for the matrix regression problem
        self.G = G
        self.W = W
        self.node_class = node_class
        self.U = U
        self.V = V
        self.T1 = T1
        self.threshold_to_add_noise = threshold_to_add_noise
        self.noise_ball_radius = noise_ball_radius
        self.num_criterion_checking_rounds = num_criterion_checking_rounds
        self.T2 = T2
        self.eta_phase_1 = eta_phase_1
        self.eta_phase_2 = eta_phase_2
        self.num_sgd_updates_per_node_per_round = num_sgd_updates_per_node_per_round
        self.phase_2_enabled = phase_2_enabled

    def _get_criterion(self, custom_nodes):
        m = len(custom_nodes.values())

        avg_grad_U = None
        avg_U = None
        avg_U_y = None
        avg_grad_V = None
        avg_V = None
        avg_V_y = None

        for node in custom_nodes.values():
            if avg_grad_U is None:
                avg_grad_U = node.U.grad.data
                avg_grad_V = node.V.grad.data
                avg_U = node.U.data
                avg_V = node.V.data
                avg_U_y = node.U_y.data
                avg_V_y = node.V_y.data
            else:
                avg_grad_U.data += node.U.grad.data
                avg_grad_V.data += node.V.grad.data
                avg_U.data += node.U.data
                avg_V.data += node.V.data
                avg_U_y.data += node.U_y.data
                avg_V_y.data += node.V_y.data
        avg_grad_U.data *= 1. / m
        avg_grad_V.data *= 1. / m
        avg_U.data *= 1. / m
        avg_V.data *= 1. / m
        avg_U_y.data *= 1. / m
        avg_V_y.data *= 1. / m

        term_1 = torch.norm(avg_grad_U, p='fro') ** 2 + torch.norm(avg_grad_V, p='fro') ** 2
        term_2 = 0
        for node in custom_nodes.values():
            term_2 += torch.norm(node.U.data - avg_U, p='fro') ** 2
            term_2 += torch.norm(node.V.data - avg_V, p='fro') ** 2
        term_2 *= 1. / m

        criterion = term_1 + term_2
        print(term_1, term_2, criterion)
        return term_1, term_2, criterion

    def run(self):
        # This is the entrypoint script

        # Now, let's create the node objects for the graph.
        custom_nodes = {node_id: self.node_class(node_id=node_id, K=len(self.G.nodes), M=self.M, rank=self.rank, W=self.W, U=self.U, V=self.V) for node_id in self.G.nodes}
        # custom_nodes = {node_id: SGDNode(node_id=node_id, K=m, M=M, rank=r, U=U, V=V) for node_id in G.nodes}

        # Now that we've created all of the nodes, let's connect them:
        for edge in self.G.edges:
            node1 = custom_nodes[edge[0]]
            node2 = custom_nodes[edge[1]]
            node1.add_neighbor(node2)
            node2.add_neighbor(node1)
        # Connect every node to every other node for the coordination step
        for node in self.G.nodes:
            custom_nodes[node].add_everyone(custom_nodes)

        # Share all the initial states with neighbors
        for node in custom_nodes.values():
            node.gather_only_x_updates_from_neighbors()
            node.gather_only_y_updates_from_everyone()
        for node in custom_nodes.values():
            node.average_y()

        # We're now ready to trigger our SGD run

        t1_idx = 0
        t2_idx = 0
        curr_iterate = 0
        in_phase_1 = True
        term_1_values = []
        term_2_values = []
        criterion_values = []
        loss_values = []
        phases = []
        criterion_checking_rounds = np.random.randint(low=0, high=int(self.T1), size=int(self.num_criterion_checking_rounds))
        while t1_idx < self.T1:
            # Optimize each node
            total_loss = 0
            for node in custom_nodes.values():
                node.optimize(num_steps=self.num_sgd_updates_per_node_per_round, step_size=self.eta_phase_1 if in_phase_1 else self.eta_phase_2)
                curr_loss = node.calculate_loss().item()
                total_loss += curr_loss
                # print("Current loss for node {} at round {} is {}".format(node.node_id, round_robin_round, curr_loss))
            print("Total loss at round {} is {}".format(curr_iterate, total_loss))
            loss_values.append(total_loss)
            # Now each node will update its own model from its neighbors
            for node in custom_nodes.values():
                node.gather_model_updates_from_neighbors()
            # Finally, apply the updates to make them permanent for each node
            for node in custom_nodes.values():
                node.apply_model_updates()

            term1, term2, criterion_at_round = self._get_criterion(custom_nodes)
            term_1_values.append(term1.item())
            term_2_values.append(term2.item())
            criterion_values.append(term1.item())
            phases.append(1 if in_phase_1 else 2)


            if in_phase_1:
                # see if we should add noise
                if self.phase_2_enabled and t1_idx in criterion_checking_rounds and criterion_at_round <= self.threshold_to_add_noise:
                    # Add noise
                    input("Prepare to add noise")
                    xi_U = torch.randn_like(custom_nodes[0].U)
                    xi_U.div_(torch.norm(xi_U, p='fro'))
                    u = np.random.uniform(0, self.noise_ball_radius)
                    xi_U.mul_(u)

                    xi_V = torch.randn_like(custom_nodes[0].V)
                    xi_V.div_(torch.norm(xi_V, p='fro'))
                    u = np.random.uniform(0, self.noise_ball_radius)
                    xi_V.mul_(u)
                    for node in custom_nodes.values():
                        # Add noise to each iterate
                        node.inject_noise(xi_U, xi_V)
                        # Recompute gradient after noise injection
                        node.reset_y()
                    for node in custom_nodes.values():
                        # Share the recomputed gradients with *everyone*, not just neighbors
                        node.gather_only_x_updates_from_neighbors()
                        node.gather_only_y_updates_from_everyone()
                    for node in custom_nodes.values():
                        # Average the new y's using *everyone's* gradients
                        node.average_y()
                    for node in custom_nodes.values():
                        # Share the recomputed gradients with *everyone*, not just neighbors
                        node.gather_only_y_updates_from_everyone()
                    # Enter phase II
                    t1_idx = 0
                    t2_idx = 0
                    in_phase_1 = False
                else:
                    t1_idx += 1
            else:
                # Don't add noise
                # check if we should remain in phase 2
                if t2_idx < self.T2:
                    t2_idx += 1
                else:
                    # Enter phase I
                    t1_idx = 0
                    t2_idx = 0
                    criterion_checking_rounds = np.random.randint(low=0,
                                                                  high=self.T1,
                                                                  size=int(self.num_criterion_checking_rounds))
                    in_phase_1 = True

            curr_iterate += 1
        input("I am about to return")
        return loss_values, term_1_values, term_2_values, criterion_values, phases

