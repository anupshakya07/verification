from gurobipy import *
import gurobipy as gb
from gurobipy import GRB
import numpy as np
import random
import math
import time
from scipy.stats import norm

import utils

LOWERBOUND = -math.log(1 + math.exp(utils.SOFTNESS * (1 - utils.THRESHOLD)))


class CitationNetworkHMLN(object):
    def __init__(self, dataset, num_nodes, num_clusters, spec_distance_matrix, spec_pred_probabilities):
        self.t_model = gb.Model(name=dataset)
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.spec_distance_matrix = spec_distance_matrix
        self.spec_pred_probabilities = spec_pred_probabilities
        self.graph, self.dataset_full = utils.load_dataset(dataset)
        self.num_classes = self.dataset_full.num_classes
        self.edges_tuple = utils.get_edges_tuple(self.graph, self.num_nodes)
        self.supernode_dict, self.node_to_cluster_idx = utils.cluster(self.num_nodes, self.num_classes,
                                                                      self.num_clusters, spec_distance_matrix)
        self.supernode_fol_dict, self.fol_node_to_cluster_idx = utils.cluster_fol(self.edges_tuple)
        self.N_fol, self.N_hfol = utils.cluster_edge_count(self.supernode_fol_dict, self.supernode_dict)
        self.supernode_fol_weights, self.supernode_hybrid_weights = utils.initialize_weights(self.num_clusters,
                                                                                             self.num_classes)

    def train(self, num_iters):
        time_1 = time.time()
        for it_num in range(num_iters):
            formula_weights, hybrid_formula_weights, dist_soft_lt_inequality, dist_soft_gt_inequality = self.initialize_distances(self.spec_distance_matrix)
            self.construct_optimization_model(formula_weights, hybrid_formula_weights)
            self.t_model.optimize()

            E_unobs_FOL, E_unobs_HFOL = self.estimate_ground_truths()

            self.t_model.remove(self.t_model.getConstrs())
            self.t_model.remove(self.t_model.getGenConstrs())
            self.t_model.update()

            for cid, edge_list in self.supernode_dict.items():
                for edge in edge_list:
                    i = edge[0]
                    j = edge[1]
                    for k in range(self.num_classes):
                        self.t_model.addConstr(
                            self.class_vars[j, k] + self.auxiliary_lt_softineq_vars[i, j, k] - self.class_vars[
                                i, k] <= 1, name="query_1_hfol_%d"%j)
                        self.t_model.addConstr(
                            self.class_vars[i, k] + self.auxiliary_lt_softineq_vars[i, j, k] - self.class_vars[
                                j, k] <= 1, name="query_2_hfol_%d"%j)
                        self.t_model.addConstr(
                            self.class_vars[j, k] - self.auxiliary_lt_softineq_vars[i, j, k] + self.class_vars[
                                i, k] <= 1, name="query_3_hfol_%d"%j)
                        self.t_model.addConstr(
                            self.class_vars[j, k] + self.auxiliary_lt_softineq_vars[i, j, k] + self.class_vars[
                                i, k] >= 1, name="query_4_hfol_%d"%j)

            ##### Adding Constraints for each link  for        N * [(Cx <-> Cy) <-> K]
            for idx, link in enumerate(self.edges_tuple):
                node_1 = link[0]
                node_2 = link[1]

                for j in range(self.num_classes):
                    self.t_model.addConstr(
                        self.class_vars[node_2, j] + self.auxiliary_vars[j, idx] - self.class_vars[node_1, j] <= 1, name="query_1_fol_%d"%j)
                    self.t_model.addConstr(
                        self.class_vars[node_1, j] + self.auxiliary_vars[j, idx] - self.class_vars[node_2, j] <= 1, name="query_2_fol_%d"%j)
                    self.t_model.addConstr(
                        self.class_vars[node_2, j] - self.auxiliary_vars[j, idx] + self.class_vars[node_1, j] <= 1, name="query_3_fol_%d"%j)
                    self.t_model.addConstr(
                        self.class_vars[node_2, j] + self.auxiliary_vars[j, idx] + self.class_vars[node_1, j] >= 1, name="query_4_fol_%d"%j)

            for i in range(self.num_nodes):
                self.t_model.addConstr(gb.quicksum(self.class_vars[i, j] for j in range(self.num_classes)) == 1,
                                       name="node_class_%d" % i)

            self.t_model.setObjective(
                gb.quicksum(
                    [gb.quicksum([self.auxiliary_vars[i, j] * formula_weights[i][j] for i in range(self.num_classes)
                                  for j, edge in enumerate(self.edges_tuple)]),
                     gb.quicksum(
                         [self.class_vars[i, j] * self.spec_pred_probabilities[i][j] for i in range(self.num_nodes)
                          for j in range(self.num_classes)]),
                     gb.quicksum([self.auxiliary_lt_softineq_vars[i, j, k] * hybrid_formula_weights[i][j][k] *
                                  dist_soft_lt_inequality[i][j][k] for i in range(self.num_nodes) for j in
                                  range(self.num_nodes) for k in range(self.num_classes)]),
                     ]),
                GRB.MAXIMIZE)
            self.t_model.update()
            self.t_model.optimize()

            E_FOL, E_HFOL = self.estimate_ground_truths()
            delta = E_unobs_FOL - E_FOL
            delta_HFOL = E_unobs_HFOL - E_HFOL

            # Delta weights for FOL
            epsilon = 0.1  # Learning Rate

            dw_fol = np.zeros(shape=(100, self.num_classes))
            dw_hfol = np.zeros(shape=(self.num_clusters, self.num_classes))
            for i in range(self.num_clusters):
                for j in range(self.num_classes):
                    if delta_HFOL[i, j] != 0:
                        dw = (epsilon * delta_HFOL[i, j]) / self.N_hfol[i]
                        dw_hfol[i, j] += dw

            for i in range(self.num_clusters):
                for j in range(self.num_classes):
                    if delta[i, j] != 0:
                        dw = (epsilon * delta[i, j]) / self.N_fol[i]
                        dw_fol[i, j] += dw

            print("DW_FOL sum = ", np.abs(dw_fol).sum())
            print("DW_HFOL sum = ", np.abs(dw_hfol).sum())
            utils.calculate_accuracy(self.class_vars, self.graph, self.num_nodes)

            if np.abs(dw_fol).sum() == 0 and np.abs(dw_hfol).sum() == 0:
                break

            self.supernode_fol_weights = self.supernode_fol_weights - dw_fol
            self.supernode_hybrid_weights = self.supernode_hybrid_weights - dw_hfol

        time_2 = time.time()
        print("Total time spent for training the specification HMLN == ", time_2 - time_1, " secs")

    def verify_nuv(self, nuv_distance_matrix, nuv_pred_probabilities):
        query_list_1, query_list_2 = utils.generate_query(self.supernode_dict)

        formula_weights, hybrid_formula_weights, dist_soft_lt_inequality, dist_soft_gt_inequality = self.initialize_distances(
            nuv_distance_matrix)
        self.construct_optimization_model(formula_weights, hybrid_formula_weights)

        self.t_model.remove(self.t_model.getConstrs())
        self.t_model.remove(self.t_model.getGenConstrs())
        self.t_model.update()

        for cid, edge_list in self.supernode_dict.items():
            for edge in edge_list:
                i = edge[0]
                j = edge[1]
                for k in range(self.num_classes):
                    self.t_model.addConstr(
                        self.class_vars[j, k] + self.auxiliary_lt_softineq_vars[i, j, k] - self.class_vars[i, k] <= 1, name="query_1_hfol_%d"%j)
                    self.t_model.addConstr(
                        self.class_vars[i, k] + self.auxiliary_lt_softineq_vars[i, j, k] - self.class_vars[j, k] <= 1, name="query_2_hfol_%d"%j)
                    self.t_model.addConstr(
                        self.class_vars[j, k] - self.auxiliary_lt_softineq_vars[i, j, k] + self.class_vars[i, k] <= 1, name="query_3_hfol_%d"%j)
                    self.t_model.addConstr(
                        self.class_vars[j, k] + self.auxiliary_lt_softineq_vars[i, j, k] + self.class_vars[i, k] >= 1, name="query_4_hfol_%d"%j)

            ##### Adding Constraints for each link  for        N * [(Cx <-> Cy) <-> K]
        for idx, link in enumerate(self.edges_tuple):
            node_1 = link[0]
            node_2 = link[1]
            #     t_model.addConstr(neighbor_vars[node_1, node_2] == 1)

            for j in range(self.num_classes):
                self.t_model.addConstr(
                    self.class_vars[node_2, j] + self.auxiliary_vars[j, idx] - self.class_vars[node_1, j] <= 1, name="query_1_fol_%d"%j)
                self.t_model.addConstr(
                    self.class_vars[node_1, j] + self.auxiliary_vars[j, idx] - self.class_vars[node_2, j] <= 1, name="query_2_fol_%d"%j)
                self.t_model.addConstr(
                    self.class_vars[node_2, j] - self.auxiliary_vars[j, idx] + self.class_vars[node_1, j] <= 1, name="query_3_fol_%d"%j)
                self.t_model.addConstr(
                    self.class_vars[node_2, j] + self.auxiliary_vars[j, idx] + self.class_vars[node_1, j] >= 1, name="query_4_fol_%d"%j)

        for i in range(self.num_nodes):
            self.t_model.addConstr(gb.quicksum(self.class_vars[i, j] for j in range(self.num_classes)) == 1,
                                   name="node_class_%d" % i)

        for q1, q2 in zip(query_list_1, query_list_2):
            self.estimate_query(q1, q2, nuv_distance_matrix, nuv_pred_probabilities, formula_weights,
                                hybrid_formula_weights, dist_soft_lt_inequality)

    def estimate_query(self, q1, q2, distance_matrix_np, pred_probabilities, formula_weights, hybrid_formula_weights,
                       dist_soft_lt_inequality):
        time_1 = time.time()
        i_1 = q1[0]
        j_1 = q1[1]

        i_2 = q2[0]
        j_2 = q2[1]

        print("Dist 1 = ", distance_matrix_np[i_1, j_1], " Dist 2 = ", distance_matrix_np[i_2, j_2])

        deg = utils.SOFTNESS * (distance_matrix_np[i_2, j_2] - distance_matrix_np[i_1, j_1])
        dist_softineq = -(math.log(1 + math.exp(deg)))

        temp = dist_soft_lt_inequality[i_2][j_2]
        dist_soft_lt_inequality[i_2][j_2] = [dist_softineq] * 7

        self.t_model.setObjective(gb.quicksum([
            gb.quicksum([self.auxiliary_vars[i, j] * formula_weights[i][j] for i in range(self.num_classes) for j, edge in
                 enumerate(self.edges_tuple)]),
            gb.quicksum([self.class_vars[i, j] * pred_probabilities[i][j] for i in range(self.num_nodes) for j in
                         range(self.num_classes)]),
            gb.quicksum([self.auxiliary_lt_softineq_vars[i, j, k] * hybrid_formula_weights[i][j][k] *
                         dist_soft_lt_inequality[i][j][
                             k] for i in range(self.num_nodes) for j in range(self.num_nodes) for k in
                         range(self.num_classes)])
        ]), GRB.MAXIMIZE)

        # C(x) == C(y) constraint
        for k in range(self.num_classes):
            self.t_model.addConstr(self.class_vars[i_2, k] == self.class_vars[j_2, k], name="query_constr_eq_%d" % k)
        self.t_model.update()

        self.t_model.optimize()
        obj_val1 = self.t_model.objVal  # Query = True Obj Value
        print("obj val 1 = ", obj_val1)

        # Removing Equality Constraints
        for k in range(self.num_classes):
            self.t_model.remove(self.t_model.getConstrByName("query_constr_eq_%d" % k))
        self.t_model.update()

        # C(x) != C(y) constraint
        for k in range(self.num_classes):
            self.t_model.addConstr(self.class_vars[i_2, k] + self.class_vars[j_2, k] <= 1,
                                   name="query_constr_neq_%d" % k)
        self.t_model.update()

        self.t_model.optimize()
        obj_val2 = self.t_model.objVal  # Query = False Obj Value

        print("obj val 2 = ", obj_val2)

        # Removing Non-equality Constraints
        for k in range(self.num_classes):
            self.t_model.remove(self.t_model.getConstrByName("query_constr_neq_%d" % k))
        self.t_model.update()

        dist_soft_lt_inequality[i_2][j_2] = temp
        time_2 = time.time()
        print(f"Time for inference = {time_2 - time_1} secs.")

    def construct_optimization_model(self, formula_weights, hybrid_formula_weights):
        self.t_model.Params.LogToConsole = 0

        self.class_vars = self.t_model.addVars(self.num_nodes, self.num_classes, vtype=GRB.BINARY, name="class_vars")
        self.auxiliary_lt_softineq_vars = self.t_model.addVars(self.num_nodes, self.num_nodes, self.num_classes,
                                                               vtype=GRB.BINARY, name="auxiliary_lt_vars")
        self.auxiliary_vars = self.t_model.addVars(self.num_classes, len(self.edges_tuple), vtype=GRB.BINARY,
                                                   name="auxiliary_vars")
        self.dist_vars = self.t_model.addVars(self.num_nodes, self.num_nodes, vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                              name="distance_vars")
        self.dist_softineq_vars = self.t_model.addVars(self.num_nodes, self.num_nodes, vtype=GRB.CONTINUOUS,
                                                       lb=LOWERBOUND,
                                                       name="distance_softineq_vars")

        ##### Adding Constraints for each pair of nodes for       Dist(x,y)<k * [(Cx <-> Cy) <-> K]
        added_edges = []
        for cid, edge_list in self.supernode_dict.items():
            for eid, edge in enumerate(edge_list):
                i = edge[0]
                j = edge[1]

                if i == j:
                    self.t_model.addConstr(self.dist_vars[i, j] == 0)
                elif str(edge) not in added_edges:
                    self.t_model.addConstr(self.dist_vars[i, j] == self.dist_vars[j, i])
                    self.t_model.addConstr(
                        self.dist_vars[i, j] <= utils.MU + utils.SIGMA * norm.ppf((eid + 2) / (len(edge_list) + 2)),
                        name="distance_lt_constr_%d_%d" % (i, j))
                    self.t_model.addConstr(
                        self.dist_vars[i, j] >= utils.MU + utils.SIGMA * norm.ppf((eid + 1) / (len(edge_list) + 2)),
                        name="distance_gt_constr_%d_%d" % (i, j))

                    added_edges.append(str(tuple((i, j))))
                    added_edges.append(str(tuple((j, i))))

                degree = self.t_model.addVar(lb=-np.inf)
                self.t_model.addConstr(degree == utils.SOFTNESS * (self.dist_vars[i, j] - utils.THRESHOLD))

                # Adding exponent of degree
                exp = self.t_model.addVar(lb=-np.inf)
                self.t_model.addGenConstrExp(degree, exp)

                # Adding 1 + exp(degree)
                z = self.t_model.addVar()
                self.t_model.addConstr(z == 1 + exp)

                # Adding log(1+ exp(degree))
                q = self.t_model.addVar(lb=-np.inf, ub=np.inf)
                self.t_model.addGenConstrLog(z, q, options="FuncPieces=-1 FuncPieceError=1")

                # Adding - log(1+exp(degree))
                self.t_model.addConstr(self.dist_softineq_vars[i, j] == -q)

                for k in range(self.num_classes):
                    self.t_model.addConstr(
                        self.class_vars[j, k] + self.auxiliary_lt_softineq_vars[i, j, k] - self.class_vars[i, k] <= 1)
                    self.t_model.addConstr(
                        self.class_vars[i, k] + self.auxiliary_lt_softineq_vars[i, j, k] - self.class_vars[j, k] <= 1)
                    self.t_model.addConstr(
                        self.class_vars[j, k] - self.auxiliary_lt_softineq_vars[i, j, k] + self.class_vars[i, k] <= 1)
                    self.t_model.addConstr(
                        self.class_vars[j, k] + self.auxiliary_lt_softineq_vars[i, j, k] + self.class_vars[i, k] >= 1)

        ##### Adding Constraints for each link  for        N * [(Cx <-> Cy) <-> K]
        for idx, link in enumerate(self.edges_tuple):
            node_1 = link[0]
            node_2 = link[1]

            for j in range(self.num_classes):
                self.t_model.addConstr(
                    self.class_vars[node_2, j] + self.auxiliary_vars[j, idx] - self.class_vars[node_1, j] <= 1)
                self.t_model.addConstr(
                    self.class_vars[node_1, j] + self.auxiliary_vars[j, idx] - self.class_vars[node_2, j] <= 1)
                self.t_model.addConstr(
                    self.class_vars[node_2, j] - self.auxiliary_vars[j, idx] + self.class_vars[node_1, j] <= 1)
                self.t_model.addConstr(
                    self.class_vars[node_2, j] + self.auxiliary_vars[j, idx] + self.class_vars[node_1, j] >= 1)

        for i in range(self.num_nodes):
            self.t_model.addConstr(gb.quicksum(self.class_vars[i, j] for j in range(self.num_classes)) == 1,
                                   name="node_class_%d" % i)

        self.t_model.ModelSense = GRB.MAXIMIZE
        self.t_model.setObjective(gb.quicksum([gb.quicksum(
            [self.auxiliary_vars[i, j] * formula_weights[i][j] for i in range(self.num_classes) for j, edge in
             enumerate(self.edges_tuple)]),
            gb.quicksum(
                [self.class_vars[i, j] * self.spec_pred_probabilities[i][j] for i in range(self.num_nodes) for
                 j in range(self.num_classes)]),
            gb.quicksum([self.auxiliary_lt_softineq_vars[i, j, k] *
                         hybrid_formula_weights[i][j][k] * self.dist_softineq_vars[i, j] for i
                         in range(self.num_nodes) for j in range(self.num_nodes) for k in
                         range(self.num_classes)])
        ]),
            GRB.MAXIMIZE)

        self.t_model.update()

    def compute_query(self, query_edge):
        i = query_edge[0]
        j = query_edge[1]

        # C(x) == C(y) constraint
        for k in range(self.num_classes):
            self.t_model.addConstr(self.class_vars[i, k] == self.class_vars[j, k], name="query_constr_eq_%d" % k)
        self.t_model.update()

        self.t_model.optimize()
        obj_val1 = self.t_model.objVal  # Query = True Obj Value
        print("obj val 1 = ", obj_val1)

        # Removing Equality Constraints
        for k in range(self.num_classes):
            self.t_model.remove(self.t_model.getConstrByName("query_constr_eq_%d" % k))
        self.t_model.update()

        # C(x) != C(y) constraint
        for k in range(self.num_classes):
            self.t_model.addConstr(self.class_vars[i, k] <= 1 - self.class_vars[j, k], name="query_constr1_neq_%d" % k)
            self.t_model.addConstr(self.class_vars[j, k] <= 1 - self.class_vars[i, k], name="query_constr2_neq_%d" % k)
        self.t_model.update()

        self.t_model.optimize()
        obj_val2 = self.t_model.objVal  # Query = False Obj Value
        print("obj val 2 = ", obj_val2)

        # Removing Non-equality Constraints
        for k in range(self.num_classes):
            self.t_model.remove(self.t_model.getConstrByName("query_constr1_neq_%d" % k))
            self.t_model.remove(self.t_model.getConstrByName("query_constr2_neq_%d" % k))
        self.t_model.update()

        return obj_val1, obj_val2

    def estimate_ground_truths(self):
        E_FOL = np.zeros(shape=(100, self.num_classes))
        for i, edge in enumerate(self.edges_tuple):
            if str(edge) in self.fol_node_to_cluster_idx:
                cid = self.fol_node_to_cluster_idx[str(edge)]
                for k in range(self.num_classes):
                    E_FOL[cid, k] += self.auxiliary_vars[k, i].X
        print("E_FOL sum = ", E_FOL.sum())

        E_HFOL = np.zeros(shape=(self.num_clusters, self.num_classes))
        for cid, edge_list in self.supernode_dict.items():
            for edge in edge_list:
                if str(edge) in self.node_to_cluster_idx:
                    i = edge[0]
                    j = edge[1]
                    for k in range(self.num_classes):
                        E_HFOL[cid, k] += self.auxiliary_lt_softineq_vars[i, j, k].X
        print("E_HFOL sum = ", E_HFOL.sum())
        return E_FOL, E_HFOL

    def initialize_distances(self, distance_matrix):
        formula_weights = []
        hybrid_formula_weights = []
        dist_soft_lt_inequality = []
        dist_soft_gt_inequality = []

        for i in range(self.num_classes):
            formula_weights_row = []
            for edge in self.edges_tuple:
                w = round(random.uniform(0, 1), 2)
                if str(edge) in self.node_to_cluster_idx:
                    cid = self.fol_node_to_cluster_idx[str(edge)]
                    w = self.supernode_fol_weights[cid][i]
                formula_weights_row.append(w)
            formula_weights.append(formula_weights_row)

        for i in range(self.num_nodes):
            temp_lt_row = []
            temp_gt_row = []
            hybrid_temp_row = []
            for j in range(self.num_nodes):

                temp_weight_row = []
                temp_lt_third_dim = []
                temp_gt_third_dim = []

                distance = distance_matrix[i, j]  # embedding distance between nodes i and j

                degree_lt = utils.SOFTNESS * (distance - utils.THRESHOLD)
                degree_gt = utils.SOFTNESS * (utils.THRESHOLD - distance)
                d_lt = -math.log(1 + math.exp(degree_lt))
                d_gt = -math.log(1 + math.exp(degree_gt))

                for k in range(self.num_classes):
                    w = 0  # round(random.uniform(0,1) ,2)
                    edge_tup = (i, j)
                    if str(edge_tup) in self.node_to_cluster_idx:
                        cid = self.node_to_cluster_idx[str(edge_tup)]
                        w = self.supernode_hybrid_weights[cid][k]
                    temp_weight_row.append(w)

                    temp_lt_third_dim.append(round(d_lt, 3))
                    temp_gt_third_dim.append(round(d_gt, 3))
                temp_lt_row.append(temp_lt_third_dim)
                temp_gt_row.append(temp_gt_third_dim)
                hybrid_temp_row.append(temp_weight_row)
            dist_soft_lt_inequality.append(temp_lt_row)
            dist_soft_gt_inequality.append(temp_gt_row)
            hybrid_formula_weights.append(hybrid_temp_row)
        return formula_weights, hybrid_formula_weights, dist_soft_lt_inequality, dist_soft_gt_inequality
