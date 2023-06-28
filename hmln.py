from gurobipy import *
import gurobipy as gb
from gurobipy import GRB
import pickle
import numpy as np
import random
import math
import time
from scipy.stats import norm
from tqdm import tqdm

import utils

LOWERBOUND = -math.log(1 + math.exp(utils.SOFTNESS * (1 - utils.THRESHOLD)))


class CitationNetworkHMLN(object):
    def __init__(self, dataset, num_nodes, num_clusters, spec_distance_matrix):
        self.supernode_dict, self.node_to_cluster_idx = utils.cluster(self.num_nodes, self.num_classes,
                                                                      self.num_clusters, spec_distance_matrix)
        self.supernode_fol_dict, self.fol_node_to_cluster_idx = utils.cluster_fol(self.edges_tuple)
        self.t_model = gb.Model(name=dataset)
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.graph, self.dataset_full = utils.load_dataset(dataset)
        self.num_classes = self.dataset_full.num_classes
        self.edges_tuple = utils.get_edges_tuple(self.graph, self.num_nodes)

    def initialize_network(self, distance_matrix):
        pass


    def construct_optimization_model(self, pred_probabilities, supernode_dict, formula_weights, hybrid_formula_weights):
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
        for cid, edge_list in supernode_dict.items():
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
                [self.class_vars[i, j] * pred_probabilities[i][j] for i in range(self.num_nodes) for
                 j in range(self.num_classes)]),
            gb.quicksum([self.auxiliary_lt_softineq_vars[i, j, k] *
                         hybrid_formula_weights[i][j][k] * self.dist_softineq_vars[i, j] for i
                         in range(self.num_nodes) for j in range(self.num_nodes) for k in
                         range(self.num_classes)]),
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

    def estimate_ground_truths(self, fol_node_to_cluster_idx, supernode_dict,
                               node_to_cluster_idx):
        E_FOL = np.zeros(shape=(100, self.num_classes))
        for i, edge in enumerate(self.edges_tuple):
            if str(edge) in fol_node_to_cluster_idx:
                cid = fol_node_to_cluster_idx[str(edge)]
                for k in range(self.num_classes):
                    E_FOL[cid, k] += self.auxiliary_vars[k, i].X
        print("E_FOL sum = ", E_FOL.sum())

        E_HFOL = np.zeros(shape=(self.num_clusters, self.num_classes))
        for cid, edge_list in supernode_dict.items():
            for edge in edge_list:
                if str(edge) in node_to_cluster_idx:
                    i = edge[0]
                    j = edge[1]
                    for k in range(self.num_classes):
                        E_HFOL[cid, k] += self.auxiliary_lt_softineq_vars[i, j, k].X
        print("E_HFOL sum = ", E_HFOL.sum())
        return E_FOL, E_HFOL

    def initialize_distances(self, supernode_fol_weights, fol_node_to_cluster_idx,
                             supernode_hybrid_weights, node_to_cluster_idx, distance_matrix):
        formula_weights = []
        hybrid_formula_weights = []
        dist_soft_lt_inequality = []
        dist_soft_gt_inequality = []

        for i in range(self.num_classes):
            formula_weights_row = []
            for edge in self.edges_tuple:
                w = round(random.uniform(0, 1), 2)
                if str(edge) in node_to_cluster_idx:
                    cid = fol_node_to_cluster_idx[str(edge)]
                    w = supernode_fol_weights[cid][i]
                formula_weights_row.append(w)
            formula_weights.append(formula_weights_row)

        # Adding constraints for linear equations generated by the formulas

        for i in range(self.num_nodes):
            temp_lt_row = []
            temp_gt_row = []
            hybrid_temp_row = []
            for j in range(self.num_nodes):

                temp_weight_row = []
                temp_lt_third_dim = []
                temp_gt_third_dim = []

                distance = distance_matrix[i][j]  # embedding distance between nodes i and j

                degree_lt = utils.SOFTNESS * (distance - utils.THRESHOLD)
                degree_gt = utils.SOFTNESS * (utils.THRESHOLD - distance)
                d_lt = -math.log(1 + math.exp(degree_lt))
                d_gt = -math.log(1 + math.exp(degree_gt))

                for k in range(self.num_classes):
                    w = 0  # round(random.uniform(0,1) ,2)
                    edge_tup = (i, j)
                    if str(edge_tup) in node_to_cluster_idx:
                        cid = node_to_cluster_idx[str(edge_tup)]
                        w = supernode_hybrid_weights[cid][k]
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
