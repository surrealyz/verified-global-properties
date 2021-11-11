#!/usr/bin/env python
# coding: utf-8

from clnmodel import CLNModel
from util import *
from gurobipy import *

class Constraint_Projector:
    def __init__(self, cln_model):
        self.grb = Model('projector')
        self.grb.setParam('Threads', 8)
        # silence console outputs
        self.grb.setParam('OutputFlag', 0)
        self.leaf_indices = cln_model.all_cids.copy()
        # add variables
        self.L = self.grb.addVars(self.leaf_indices, vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, name = 'l')
        self.diff_list = []
        self.absdiff_list = []
        return

    def update(self, cln_model):
        # add variables for new leaves
        for idx in cln_model.all_cids:
            if idx not in self.leaf_indices:
                self.L[idx] = self.grb.addVar(vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, name = 'l')
        self.grb.update()
        self.leaf_indices = cln_model.all_cids.copy()
        return

    def clear(self):
        # remove all existing variables and constraints
        self.grb.remove(self.grb.getVars())
        self.grb.remove(self.grb.getConstrs())
        self.grb.remove(self.grb.getGenConstrs())
        return

    def add_constr_sum_le_sum(self, before_true_cids, after_true_cids):
        before_vars = [self.L[ref_cid] for ref_cid in before_true_cids]
        after_vars = [self.L[ref_cid] for ref_cid in after_true_cids]
        self.grb.addConstr(LinExpr([1]*len(before_vars), before_vars) <= LinExpr([1]*len(after_vars), after_vars))
        self.grb.update()
        return

    def add_constr_absdiff_le_c(self, before_true_cids, after_true_cids, output_diff):
        diff = self.grb.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
        absdiff = self.grb.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
        self.diff_list.append(diff)
        self.absdiff_list.append(absdiff)
        before_vars = [self.L[ref_cid] for ref_cid in before_true_cids]
        after_vars = [self.L[ref_cid] for ref_cid in after_true_cids]
        self.grb.addConstr(diff == LinExpr([1]*len(before_vars), before_vars) - \
                            LinExpr([1]*len(after_vars), after_vars))
        self.grb.addConstr(absdiff == abs_(diff))
        self.grb.addConstr(absdiff <= output_diff)
        self.grb.update()
        return

    def add_constr_diff_le_c(self, before_true_cids, after_true_cids, output_diff, cname):
        diff = self.grb.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
        before_vars = [self.L[ref_cid] for ref_cid in before_true_cids]
        after_vars = [self.L[ref_cid] for ref_cid in after_true_cids]
        self.grb.addConstr(diff == LinExpr([1]*len(before_vars), before_vars) - \
                            LinExpr([1]*len(after_vars), after_vars), name = cname)
        self.grb.addConstr(diff <= output_diff)
        self.grb.update()
        return

    def add_constr_sum_ge_c(self, select_cids, const):
        select_vars = [self.L[ref_cid] for ref_cid in select_cids]
        self.grb.addConstr(LinExpr([1]*len(select_vars), select_vars) >= const)
        self.grb.update()
        return

    def project(self, in_leaves):
        expr = 0
        for cid, leaf_val in in_leaves.items():
            expr += (leaf_val - self.L[cid]) * (leaf_val - self.L[cid])
        self.grb.setObjective(expr, GRB.MINIMIZE)
        # optimize includes update, unless we want to print something before that.
        self.grb.optimize()
        if self.grb.Status == GRB.OPTIMAL:
            #print('projection model was optimally solved\n')
            pass
        elif self.grb.Status == GRB.INFEASIBLE:
            print('projection model was infeasible\n')
            exit()
        else:
            #print('projection model status code:', self.grb.Status)
            pass

        #print('after projection')
        #for v in self.grb.getVars():
        #    print('%s %g' % (v.varName, v.x))
        out_leaves = {}
        for cid in self.leaf_indices:
            out_leaves[cid] = self.L[cid].x
        return out_leaves
