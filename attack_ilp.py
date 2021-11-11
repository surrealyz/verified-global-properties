#!/usr/bin/env python
# coding: utf-8

import sys
import argparse
import copy
import xgboost as xgb
import numpy as np
import json

import math
from collections import defaultdict
import random
import torch
from clnmodel import CLNModel
from util import *

from gurobipy import *

# if min difference between feature values < GUARD_VAL
# this needs to be smaller, or round features
GUARD_VAL = 1e-6
DIFF_VAL = 5e-5
#DIFF_VAL = 1e-6
ROUND_DIGITS = 6
#ROUND_DIGITS = 12
MAX_INT32 = pow(2, 31)-1

def parse_args():
    parser = argparse.ArgumentParser(description='ILP formulation to check global model property.')
    parser.add_argument('--model_type', type=str, choices=['cln', 'xgboost'], help='choose the type of model to attack.', required=True)
    parser.add_argument('--model_path', type=str, help='cln: model path. xgboost: input json file name.', required=False)
    parser.add_argument('--nfeat', type=int, default=None, help='number of features.', required=True)
    parser.add_argument('--nlabels', type=int, default=None, help='number of labels.', required=True)
    parser.add_argument('--intfeat', type=str, default=None, help='clip adv feature values to integers according to the config file.')
    parser.add_argument('--default_lo', type=float, default=None, help='default lower bound for features', required=False)
    parser.add_argument('--default_hi', type=float, default=MAX_INT32, help='default upper bound for features', required=False)
    parser.add_argument('--monotonicity', type=str, default=None, help='list of feature indices for the monotone features.', required=False)
    parser.add_argument('--monotonicity_dir', type=str, default=None, help='the direction of monotone features.', required=False)
    parser.add_argument('--stability', type=str, default=None, help='list of feature indices for the stable features.', required=False)
    parser.add_argument('--stability_th', type=float, default=0.2, help='the constant threshold for stable features.', required=False)
    parser.add_argument('--eps', type=float, default=None, help='input bound for small modification distance property.', required=False)
    parser.add_argument('--C', type=float, default=None, help='lipschitz constant for small modification distance property.', required=False)
    parser.add_argument('--featmax', type=str, help='csv file for max feature values.', required=False)
    parser.add_argument('--lowcost_th', type=float, default=0.98, help='the confidence threshold of low cost features.', required=False)
    parser.add_argument('--lowcost', type=str, default=None, help='dict of feature bounds for the low cost feature property.', required=False)
    parser.add_argument('--redundancy', type=str, default=None, help='sets of features for redundancy. \
                        e.g., [{0:(6, None), 1:(None, None)}, {8:(None, None), 9:(None, None)}, \
                            {10:(None, None), 11:(None, None)}, {12:(None, None), 13:(None, None)}] \
                            says that [0, 1], [8, 9], [10, 11], [12, 13] are redundant of each other.')
    parser.add_argument('--start_cid', type=int, default=-1, help='the start cid to perform attack.', required=False)
    parser.add_argument('--end_cid', type=int, default=-1, help='the end cid to perform attack.', required=False)
    parser.add_argument('--int_var', action='store_true', default=False, required=False)
    parser.add_argument('--no_timeout', action='store_true', default=False, required=False)
    parser.add_argument('--test', '--test_data', type=str, help='test data file name.', required=False)
    parser.add_argument('--offset', type=int, default=0, help='start index to do local attack.', required=False)
    return parser.parse_args()

class ILPModel(object):
    def __init__(self, cln_model, start_cid, end_cid, int_indices, nfeat, default_lo,
                guard_val=GUARD_VAL, round_digits=ROUND_DIGITS, binary=True):
        self.cln_model = cln_model
        self.int_indices = int_indices
        self.nfeat = nfeat
        self.default_lo = default_lo
        self.guard_val = guard_val
        self.round_digits = round_digits
        self.binary = binary
        self.leaf_indices = list(range(start_cid, end_cid + 1))
        self.second_leaf_indices = [cid+self.cln_model.last_cid+1 for cid in self.leaf_indices]
        self.grb = Model('attack')
        self.grb.setParam('Threads', 4)
        # silence console outputs
        self.grb.setParam('OutputFlag', 0)
        # Integer feasibility tolerance
        #self.grb.setParam('IntFeasTol', 1e-9)
        # the solver work harder to try to avoid solutions that exploit integrality tolerances
        #self.grb.setParam('IntegralityFocus', 1)
        self.grb.setParam('TimeLimit', 30)
        return

    # not calling this anymore.
    def update(self, cln_model):
        #self.cln_model = cln_model
        self.leaf_indices = self.cln_model.all_cids
        self.second_leaf_indices = [cid+self.cln_model.last_cid+1 for cid in self.cln_model.all_cids]
        return

    def global_attack(self, start_cid, end_cid, prop_name, *args):
        # activation variables. double them.
        self.L = self.grb.addVars(self.leaf_indices + self.second_leaf_indices, \
                vtype = GRB.BINARY, lb = 0, ub = 1, name = 'l')

        # from leaf var (or 2nd set) to activation value
        self.leaf_value = {}

        # feature index that we need to assign different predicate variables
        fi_list = args[0]
        print('\n*** feature index %s, cid from %d to %d ***' % (fi_list, start_cid, end_cid))

        # from lid to list of predicate variables
        self.lid_vars = defaultdict(list)
        # from lid, aid, to predicate variable
        self.atom_var = {}
        self.atom_var_neg = {}

        # from predicate variable, to (feature index, threshold)
        self.pinfo = {}
        # p dictionary by attributes, {attr1:[(threshold1, gurobiVar1),(threshold2, gurobiVar2),...],attr2:[...]}
        self.pdict = defaultdict(list)
        # track identical thresholds for the same attribute
        self.pcheck = {}

        changed_atoms = defaultdict(list)
        pred_scores = None

        for jj in range(2):
            # read each clause and create predicate variables as needed
            # use (cid, aid) to index the predicate variable.
            # add AND constraint
            for cid in range(start_cid, end_cid + 1):
                lid = cid + jj*(self.cln_model.last_cid + 1)
                for aid in range(self.cln_model.clause_atom_cnt[cid]):
                    # check if it's negating another clause
                    src_cid = self.cln_model.negate_clause_src.get('%s_%s' % (cid, aid), None)
                    # negate an existing atom
                    neg_cid, neg_aid = self.cln_model.negate_atom_src.get('%s_%s' % (cid, aid), (None, None))
                    # same as an existing atom
                    same_cid, same_aid = self.cln_model.same_atom_src.get('%s_%s' % (cid, aid), (None, None))
                    if src_cid != None:
                        # this is never executed in the current version.
                        qname = 'q_%s_%s' % (cid, aid)
                        # negating an entire clause
                        self.atom_var[lid, aid] = self.grb.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name = qname)
                        self.grb.addConstr(self.atom_var[lid, aid] + self.L[src_cid + jj*(self.cln_model.last_cid + 1)] == 1, name = qname)
                        self.grb.update()
                    elif neg_aid != None:
                        # double the variables related to fi
                        # if we always negate in this form not ( x < threshold), then it should be q variables
                        wi = 'w_%s_%s' % (neg_cid, neg_aid)
                        feat_idx = self.cln_model.atom_feat[wi]
                        if feat_idx not in fi_list:
                            qname = 'q_%s_%s' % (cid, aid)
                        else:
                            qname = 'q_%s_%s_j%s' % (cid, aid, jj)
                        # negate an existing atom
                        self.atom_var[lid, aid] = self.grb.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name = qname)
                        self.grb.addConstr(self.atom_var[lid, aid] == 1 - self.atom_var[neg_cid + jj*(self.cln_model.last_cid + 1), neg_aid], name = qname)
                        self.grb.update()
                    elif same_aid != None:
                        # double the variables related to fi
                        # if we always negate in this form not ( x < threshold), then it should be q variables
                        wi = 'w_%s_%s' % (same_cid, same_aid)
                        feat_idx = self.cln_model.atom_feat[wi]
                        if feat_idx not in fi_list:
                            qname = 'q_%s_%s' % (cid, aid)
                        else:
                            qname = 'q_%s_%s_j%s' % (cid, aid, jj)
                        # same as an existing atom
                        self.atom_var[lid, aid] = self.grb.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name = qname)
                        self.grb.addConstr(self.atom_var[lid, aid] == self.atom_var[same_cid + jj*(self.cln_model.last_cid + 1), same_aid], name = qname)
                        self.grb.update()
                    else:
                        # get w and eta for the atom
                        wi = 'w_%s_%s' % (cid, aid)
                        ei = 'eta_%s_%s' % (cid, aid)
                        weights = self.cln_model.params[wi].data.cpu().numpy()
                        eta = self.cln_model.params[ei].item()

                        cmp_name = self.cln_model.atom_cmp[cid][aid]
                        # we work with single feature atoms for now
                        feat_idx = self.cln_model.atom_feat[wi]
                        weight = weights[0]


                        # double the variables related to fi
                        if feat_idx not in fi_list:
                            pname = 'p_%s_%s' % (cid, aid)
                            qname = 'q_%s_%s' % (cid, aid)
                        else:
                            pname = 'p_%s_%s_j%s' % (cid, aid, jj)
                            qname = 'q_%s_%s_j%s' % (cid, aid, jj)

                        p_negate = False
                        if feat_idx in fi_list or jj != 1:
                            # add predicate variables and negation constraint
                            # according to the inequality, decide whether add another predicate variable
                            if weight != 0:
                                if weight > 0 and cmp_name == '<':
                                    threshold = eta / weight
                                    ### p_negate = False
                                elif weight < 0 and cmp_name == '>=':
                                    threshold = eta / weight + self.guard_val
                                    ### p_negate = False
                                elif weight > 0 and cmp_name == '>=':
                                    threshold = eta / weight
                                    p_negate = True
                                    ### p_negate = True
                                else:
                                    # weight < 0 and cmp_name == '<'
                                    threshold = eta / weight + self.guard_val
                                    p_negate = True
                            else:
                                # use np.inf and -np.inf to track threshold, pinfo, pdict
                                if (eta >= 0 and cmp_name == '<') or (eta == 0 and cmp_name == '>=') or (eta < 0 and cmp_name == '>='):
                                    threshold = np.inf
                                else:
                                    threshold = -np.inf
                                if feat_idx in self.int_indices:
                                    tup_th = math.ceil(threshold)
                                else:
                                    tup_th = threshold
                                if feat_idx in fi_list:
                                    tup = feat_idx, tup_th, jj
                                else:
                                    tup = feat_idx, tup_th
                                if tup not in self.pcheck:
                                    if threshold == np.inf:
                                        self.pcheck[tup] = self.grb.addVar(lb = 1, ub = 1, vtype = GRB.BINARY, name = pname)
                                    if threshold == -np.inf:
                                        self.pcheck[tup] = self.grb.addVar(lb = 0, ub = 0, vtype = GRB.BINARY, name = pname)
                                self.atom_var[lid, aid] = self.pcheck[tup]
                            if feat_idx in self.int_indices:
                                tup_th = math.ceil(threshold)
                            else:
                                tup_th = threshold
                            if feat_idx in fi_list:
                                tup = feat_idx, tup_th, jj
                            else:
                                tup = feat_idx, tup_th

                            if p_negate == False:
                                # for each predicate, we need:
                                # attribute/feature index, computed threshold (x_j < eta), inequality
                                if weight != 0:
                                    if tup not in self.pcheck:
                                        self.pcheck[tup] = self.grb.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name = pname)
                                        # feature value >= self.default_lo, x_j < self.default_lo is always false.
                                        if self.default_lo != None and threshold <= self.default_lo:
                                            pvar = self.pcheck[tup] 
                                            self.grb.addConstr(pvar <= 0, name = 'redundant_%s' % pname)
                                        if feat_idx not in fi_list:
                                            self.pdict[feat_idx].append((threshold, self.pcheck[tup]))
                                        else:
                                            self.pdict['%s_%s' % (feat_idx, jj)].append((threshold, self.pcheck[tup]))
                                    self.atom_var[lid, aid] = self.pcheck[tup]
                                self.pinfo[pname] = (feat_idx, threshold, cmp_name)
                            else:
                                # p_negate = True
                                if tup not in self.pcheck:
                                    self.pcheck[tup] = self.grb.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name = pname)
                                    # feature value >= self.default_lo, x_j < self.default_lo is always false.
                                    if self.default_lo != None and threshold <= self.default_lo:
                                        pvar = self.pcheck[tup]
                                        self.grb.addConstr(pvar <= 0, name = 'redundant_%s' % pname)
                                    if feat_idx not in fi_list:
                                        self.pdict[feat_idx].append((threshold, self.pcheck[tup]))
                                    else:
                                        self.pdict['%s_%s' % (feat_idx, jj)].append((threshold, self.pcheck[tup]))
                                self.atom_var_neg[lid, aid] =  self.pcheck[tup]
                                # always add a new q variable
                                self.atom_var[lid, aid] = self.grb.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name = qname)
                                # p = 1 - q
                                self.grb.addConstr(self.atom_var[lid, aid] + self.atom_var_neg[lid, aid] == 1, name = 'neg_p_%s_%s' % (lid, aid))
                                self.pinfo[pname] = (feat_idx, threshold, cmp_name)

                            self.grb.update()
                        else:
                            self.atom_var[lid, aid] = self.atom_var[cid, aid]

                    if feat_idx not in fi_list:
                        # always use the first set of variables if it's not the perturbed feature
                        self.lid_vars[lid].append(self.atom_var[cid, aid])
                    else:
                        self.lid_vars[lid].append(self.atom_var[lid, aid])
                    self.grb.update()

                if cid not in self.cln_model.del_cids:
                    # get leaf for the clause
                    li = 'l_%s' % cid
                    leaf_val = self.cln_model.params[li].item()
                    #print(lid)
                    #print(self.lid_vars[lid])
                    # AND(all predicates for the clause) = self.L[lid]
                    self.grb.addConstr(LinExpr([1]*len(self.lid_vars[lid]), self.lid_vars[lid]) - len(self.lid_vars[lid])*self.L[lid] >= 0, name = 'conjunction_%s_1' % lid)
                    self.grb.addConstr(LinExpr([1]*len(self.lid_vars[lid]), self.lid_vars[lid]) - len(self.lid_vars[lid])*self.L[lid] <= len(self.lid_vars[lid])-1, name = 'conjunction_%s_2' % lid)

                    # keep track of leaf_val * L[lid]
                    self.leaf_value[lid] = leaf_val
                else:
                    self.grb.addConstr(self.lid_vars[lid] <= 0)
                    self.leaf_value[lid] = 0


        if prop_name == 'eps':
            eps = args[1]
            featmax = args[3]
            # add auxiliary predicates to capture linf distances
            self.auxdict = copy.copy(self.pdict)
            self.range_vars = defaultdict(list)
            j = 0
            for key in sorted(self.pdict.keys()):
                # we will also have sorted pdict
                self.pdict[key].sort(key=lambda tup: tup[0])
                # add threshold - eps, threshold + eps variables.
                # bound both fid_0 and fid_1 cases
                fid, jj = key.split('_')
                fid = int(fid)
                fid_eps = featmax[fid] * eps
                if jj == '0':
                    for i in range(len(self.pdict[key])-1):
                        th = self.pdict[key][i][0]
                        var1 = self.grb.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name = 'a_%d' % j)
                        self.auxdict[key].append((th - fid_eps, var1))
                        j += 1
                        
                        var2 = self.grb.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name = 'a_%d' % j)
                        self.auxdict[key].append((th + fid_eps, var2))
                        j += 1

                        bvar = self.grb.addVar(lb = 0, ub = 1, vtype = GRB.INTEGER)
                        self.range_vars[fid].append(bvar)
                        self.grb.addConstr(bvar == var2 - var1)

                        key2 = '%d_1' % fid
                        self.auxdict[key2].append((th - fid_eps, var1))
                        self.auxdict[key2].append((th + fid_eps, var2))
            
            for key in self.auxdict.keys():
                min_diff = 1000
                if len(self.auxdict[key])>1:
                    # sort all predicates including auxiliary variables
                    self.auxdict[key].sort(key=lambda tup: tup[0])
                    for i in range(len(self.auxdict[key])-1):
                        self.grb.addConstr(self.auxdict[key][i][1]<=self.auxdict[key][i+1][1], name='a_consis_attr{}_{}th'.format(key,i))
                        # if it is an integer feature
                        if (key in self.int_indices or \
                            (type(key)==str and int(key.split('_')[0]) in self.int_indices)) \
                            and math.floor(self.auxdict[key][i][0]) == math.floor(self.auxdict[key][i+1][0]):
                            # add integer p consistency constraints
                            self.grb.addConstr(self.auxdict[key][i][1]==self.auxdict[key][i+1][1], name='a_int_consis_attr{}_{}th'.format(key,i))
                        else:
                            # get min_diff for float feaures
                            min_diff = min( min_diff, self.pdict[key][i+1][0]-self.pdict[key][i][0])
                    print('attr {} min difference between thresholds:{}'.format(key,min_diff))
        
        else:
            # sort each feature list
            # add p constraints
            for key in self.pdict.keys():
                min_diff = 1000
                if len(self.pdict[key])>1:
                    self.pdict[key].sort(key=lambda tup: tup[0])
                    for i in range(len(self.pdict[key])-1):
                        self.grb.addConstr(self.pdict[key][i][1]<=self.pdict[key][i+1][1], name='p_consis_attr{}_{}th'.format(key,i))
                        # if it is an integer feature
                        if (key in self.int_indices or \
                            (type(key)==str and int(key.split('_')[0]) in self.int_indices)) \
                            and math.floor(self.pdict[key][i][0]) == math.floor(self.pdict[key][i+1][0]):
                            # add integer p consistency constraints
                            self.grb.addConstr(self.pdict[key][i][1]==self.pdict[key][i+1][1], name='p_int_consis_attr{}_{}th'.format(key,i))
                        else:
                            # get min_diff for float feaures
                            min_diff = min( min_diff, self.pdict[key][i+1][0]-self.pdict[key][i][0])
                    print('attr {} min difference between thresholds:{}'.format(key,min_diff))
                    #if min_diff < 2 * self.guard_val:
                    #    self.guard_val = min_diff/3
                    #    print('guard value too large, change to min_diff/3:',self.guard_val)

        x1_pvars = []
        x2_pvars = []
        for key, varlist in self.pdict.items():
            for threshold, grbvar in varlist:
                if type(key) != str:
                    x1_pvars.append(grbvar)
                    x2_pvars.append(grbvar)
                elif key.split('_')[-1] == '0':
                    x1_pvars.append(grbvar)
                else:
                    x2_pvars.append(grbvar)
        #print(x1_pvars)
        #print(x2_pvars)

        leaf_values = [self.leaf_value[lid] for lid in self.leaf_indices]
        first_llist = [self.L[lid] for lid in self.leaf_indices]
        second_llist = [self.L[lid] for lid in self.second_leaf_indices]

        # prop_name
        if prop_name == 'monotonicity':
            direction = args[1]
            # x1 < x2. more predicates are true in x1.
            self.grb.addConstr(LinExpr([1]*len(x1_pvars), x1_pvars) >= LinExpr([1]*len(x2_pvars), x2_pvars) + 1, name = 'input')
            # f(x1) > f(x2)
            if direction == 1:
                self.grb.addConstr(LinExpr(leaf_values, first_llist) >= LinExpr(leaf_values, second_llist) + DIFF_VAL, name = 'output')
            elif direction == -1:
                self.grb.addConstr(LinExpr(leaf_values, first_llist) + DIFF_VAL <= LinExpr(leaf_values, second_llist), name = 'output')

        elif prop_name == 'stability':
            maxdiff = args[1]
            # x1 != x2: 1 - (x1 == x2)
            in_diff = self.grb.addVar(lb = -GRB.INFINITY, vtype = GRB.INTEGER, name = 'in_diff')
            abs_in_diff = self.grb.addVar(vtype = GRB.INTEGER, name = 'abs_in_diff')
            self.grb.addConstr(in_diff == LinExpr([1]*len(x1_pvars), x1_pvars) - LinExpr([1]*len(x2_pvars), x2_pvars), name = 'input_diff')
            self.grb.addConstr(abs_in_diff == abs_(in_diff))
            # at least one predicate is different
            self.grb.addConstr(abs_in_diff >= 1)

            out_diff = self.grb.addVar(lb= -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'out_diff')
            abs_out_diff = self.grb.addVar(vtype = GRB.CONTINUOUS, name = 'abs_out_diff')
            self.grb.addConstr(out_diff == (LinExpr(leaf_values, first_llist) - LinExpr(leaf_values, second_llist)), name = 'output_diff')
            self.grb.addConstr(abs_out_diff == abs_(out_diff))
            self.grb.addConstr(abs_out_diff >= maxdiff + DIFF_VAL)
        
        elif prop_name == 'eps':
            eps = args[1]
            maxdiff = args[2]
            
            # input constraints: same eps range for each feature dimension of x and x'
            # for each feature dimension
            # for each interval
            # do the or constraints for all intervals
            for fidx in self.range_vars.keys():
                # OR
                # l1 - u1 OR l2 - u2, ...
                fid_n = len(self.range_vars[fidx])
                self.grb.addConstr(fid_n - LinExpr([1]*fid_n, self.range_vars[fidx]) >= 0, name = 'range_or_%s_1' %fidx)
                self.grb.addConstr(fid_n - LinExpr([1]*fid_n, self.range_vars[fidx]) <= fid_n - 1, name = 'range_or_%s_2' %fidx)
            
            # how does var2 - var1 affect int features?
            # if floor(var2 th) == floor(var1 th), var2 - var1 == 0

            out_diff = self.grb.addVar(lb= -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'out_diff')
            abs_out_diff = self.grb.addVar(vtype = GRB.CONTINUOUS, name = 'abs_out_diff')
            self.grb.addConstr(out_diff == (LinExpr(leaf_values, first_llist) - LinExpr(leaf_values, second_llist)), name = 'output_diff')
            self.grb.addConstr(abs_out_diff == abs_(out_diff))
            self.grb.addConstr(abs_out_diff >= maxdiff + DIFF_VAL)
            
        elif prop_name == 'lowcost':
            lower = args[1]
            upper = args[2]
            cutoff = args[3]
            confidence = args[4]
            # self.pdict['%s_%s' % (feat_idx, jj)].append((threshold, pvar))
            # assume there is only one key for lowcost property
            key = '%s_1' % fi_list[0]
            if lower != None or upper != None:
                for threshold, pvar in self.pdict[key]:
                    if lower != None and threshold <= lower:
                        self.grb.addConstr(pvar == 0, name = 'fid_lower')
                    if upper != None and threshold > upper:
                        self.grb.addConstr(pvar == 1, name = 'fid_upper')
            before_logit = inv_sigmoid(confidence)
            after_logit = inv_sigmoid(cutoff)
            self.grb.addConstr(LinExpr(leaf_values, first_llist) >= before_logit, name = 'precondition')
            self.grb.addConstr(LinExpr(leaf_values, second_llist) <= after_logit - DIFF_VAL, name = 'postcondition')
        
        elif prop_name == 'redundancy':
            lower_list = args[1]
            upper_list = args[2]
            cutoff = args[3]
            confidence = args[4]
            for i, fi in enumerate(fi_list):
                key = '%s_1' % fi
                lower = lower_list[i]
                upper = upper_list[i]
                if lower != None or upper != None:
                    for threshold, pvar in self.pdict[key]:
                        if lower != None and threshold <= lower:
                            self.grb.addConstr(pvar == 0)
                        if upper != None and threshold > upper:
                            self.grb.addConstr(pvar == 1)
            before_logit = inv_sigmoid(confidence)
            after_logit = inv_sigmoid(cutoff)
            self.grb.addConstr(LinExpr(leaf_values, first_llist) >= before_logit, name = 'precondition')
            self.grb.addConstr(LinExpr(leaf_values, second_llist) <= after_logit - DIFF_VAL, name = 'postcondition')
        
        else:
            # implement other properties
            print("property %s not implemented yet" % prop_name)
            raise SystemExit
            return

        # no objective
        self.grb.setObjective(0, GRB.MINIMIZE)
        self.grb.update()
        # debug
        #self.grb.write('debugmodel.lp')
        print("*** DEBUG: starting the attack")
        sys.stdout.flush()
        self.grb.optimize()
        status = self.grb.Status
        if status == GRB.OPTIMAL:
            print('model was optimally solved\n')
            sys.stdout.flush()
            # for v in self.grb.getVars():
            #     print('%s %g' % (v.varName, v.x))

            # generate counterexamples x1 and x2
            # so we can use the model wrapper to check whether the output is expected
            x1 = [0.0 for cnt in range(self.nfeat)]
            x2 = [0.0 for cnt in range(self.nfeat)]
            for key in self.pdict.keys():
                #print(key, self.pdict[key])
                if type(key) != str:
                    idx = key
                    if idx in self.int_indices:
                        isint = True
                    else:
                        isint = False
                    val = self.genval(self.pdict[key], isint)
                    x1[idx] = val
                    x2[idx] = val
                else:
                    idx = int(key.split('_')[0])
                    if idx in self.int_indices:
                        isint = True
                    else:
                        isint = False
                    val = self.genval(self.pdict[key], isint)
                    if key.split('_')[-1] == '0':
                        x1[idx] = val
                        #print('x1', self.pdict[key])
                    else:
                        x2[idx] = val
                        #print('x2', self.pdict[key])
            print('\n*** counterexample ***')
            print('x1:', x1)
            print('weighted_sum_of_leaves:', sum([self.leaf_value[lid]*self.L[lid].x for lid in self.leaf_indices]))
            print('x2:', x2)
            print('weighted_sum_of_leaves:', sum([self.leaf_value[lid]*self.L[lid].x for lid in self.second_leaf_indices]), '\n')

            # print out atom state and clause state changes. oracle result
            for cid in self.leaf_indices:
                lid_1 = cid
                lid_2 = cid + self.cln_model.last_cid + 1
                lvar_1 = self.L[lid_1]
                lvar_2 = self.L[lid_2]
                if lvar_1.x != lvar_2.x:
                    print('cid', cid, 'changed from', lvar_1.x, 'to', lvar_2.x)
                    for aid in range(self.cln_model.clause_atom_cnt[cid]):
                        if self.atom_var[lid_1, aid].x != self.atom_var[lid_2, aid].x:
                            print('cid', cid, 'aid', aid, 'changed from', \
                            self.atom_var[lid_1, aid].x, 'to', self.atom_var[lid_2, aid].x)

            # see what atoms changed states
            atom_1, clause_1, leaf_ret_1 = check_states(self.cln_model, start_cid, end_cid, self.int_indices, x1)
            atom_2, clause_2, leaf_ret_2 = check_states(self.cln_model, start_cid, end_cid, self.int_indices, x2)
            pred_scores = [leaf_ret_1, leaf_ret_2]
            for cid in leaf_ret_1.keys():
                if clause_1[cid] == clause_2[cid]:
                    continue
                # compare the atoms
                for aid in range(len(atom_1[cid])):
                    if atom_1[cid][aid] != atom_2[cid][aid]:
                        changed_atoms[cid].append((aid, atom_2[cid][aid]))
            print('changed_atoms:', changed_atoms)
            before_score = 0
            after_score = 0
            # before_true_cids = set()
            # after_true_cids = set()
            # get after_true_cids from everything, not just cids that changed states
            for ref_cid, all_scores in pred_scores[1].items():
                if pred_scores[0][ref_cid][0] != 0 and pred_scores[0][ref_cid][0] != None:
                    # before_true_cids.add(ref_cid)
                    before_score += pred_scores[0][ref_cid][0]
                if all_scores[0] != 0 and all_scores[0] != None:
                    # after_true_cids.add(ref_cid)
                    after_score += all_scores[0]
            
            print('before_score:', before_score, 'after_score:', after_score)
            # before_pred = self.cln_model.discrete_logits(torch.from_numpy(np.array([x1])).float(), [None])
            # after_pred = self.cln_model.discrete_logits(torch.from_numpy(np.array([x2])).float(), [None])
            # # if we change part of the model, other parts of the model may have atoms changed too.
            # print('before_pred:', before_pred, 'after_pred:', after_pred)
        elif status == GRB.INFEASIBLE:
            print('model was infeasible\n')
        else:
            print('status code:', status)
        sys.stdout.flush()
        print('*** DEBUG: before trying to remove stuff for grb')
        sys.stdout.flush()
        # remove all existing variables and constraints
        self.grb.remove(self.grb.getConstrs())
        self.grb.update()
        self.grb.remove(self.grb.getGenConstrs())
        self.grb.remove(self.grb.getVars())
        self.grb.update()

        return changed_atoms, pred_scores, status


    def genval(self, tuple_list, isint):
        # tuple_list sorted by the threshold
        last = self.default_lo
        cur = self.default_lo
        sample = None
        for threshold, pvar in tuple_list:
            # get the solved value from pvar
            cur = threshold
            if pvar.x == 1:
                # sample between [last, cur)
                if isint == True:
                    sample = random.randint(math.ceil(last), math.floor(cur))
                else:
                    sample = random.uniform(last, cur-GUARD_VAL)
                return sample
            last = cur
        if isint == True:
            sample = random.randint(math.ceil(last), MAX_INT32)
        else:
            sample = random.uniform(last, float(MAX_INT32))
        return sample

    def changeval(self, val, tuple_list, isint):
        last = self.default_lo
        cur = self.default_lo
        for threshold, pvar in tuple_list:
            # get the solved value from pvar
            cur = threshold
            if pvar.x > 0.5 and val >= cur:
                sample = cur - self.guard_val
                if isint == True:
                    sample = math.floor(sample)
                return sample
            if pvar.x < 0.5 and val < cur:
                sample = cur + self.guard_val
                if isint == True:
                    sample = math.ceil(sample)
                return sample
            last = cur
        return val

def main(args):

    if args.intfeat != None:
        infile = json.load(open(args.intfeat, 'r'))
        int_indices = infile['indices']

    if args.model_type == 'cln':
        # read the pytorch model
        cln_model = torch.load(args.model_path)
        print("cln model loaded from", args.model_path)
        #cln_model.prettyprint()
    elif args.model_type == 'xgboost':
        cln_model = CLNModel(None, 0, 1, 1, \
                            args.model_path, args.model_path, -1, True, True, int_indices, args.nfeat, args.nlabels, \
                            [], negate=False)
        cln_model = cln_model.cuda()
    else:
        exit()

    st = args.start_cid if args.start_cid != -1 else 0
    ed = args.end_cid if args.end_cid != -1 else cln_model.last_cid
    attack_model = ILPModel(cln_model, st, ed, int_indices, args.nfeat, args.default_lo)

    if args.int_var:
        attack_model.grb.setParam('IntFeasTol', 1e-9)
    
    if args.no_timeout:
        attack_model.grb.setParam('TimeLimit', GRB.INFINITY)

    if args.monotonicity != None:
        monotone_index_list = eval(args.monotonicity)
        monotone_direction = eval(args.monotonicity_dir)
        for i, fi in enumerate(monotone_index_list):
            direction = monotone_direction[i]
            attack_model.global_attack(st, ed, 'monotonicity', [fi], direction)
    elif args.stability != None:
        stable_index_list = eval(args.stability)
        stable_threshold = args.stability_th
        for i, fi in enumerate(stable_index_list):
            attack_model.global_attack(st, ed, 'stability', [fi], stable_threshold)
    elif args.eps != None:
        featmax = np.loadtxt(args.featmax, delimiter=',', usecols=list(range(args.nfeat))).astype(np.float32)
        constant = args.C
        eps = args.eps
        maxdiff = eps * constant
        attack_model.global_attack(st, ed, 'eps', list(range(args.nfeat)), eps, maxdiff, featmax)
    elif args.lowcost != None:
        lowcost_dict = eval(args.lowcost)
        confidence_threshold = args.lowcost_th
        cutoff = 0.5
        for lowcost_index, bounds in lowcost_dict.items():
            lower, upper = bounds
            attack_model.global_attack(st, ed, 'lowcost', [lowcost_index], lower, upper, cutoff, confidence_threshold)
    elif args.redundancy != None:
        lowcost_array = eval(args.redundancy)
        confidence_threshold = args.lowcost_th
        cutoff = 0.5
        for lowcost_dict in lowcost_array:
            fi_list = []
            lower_list = []
            upper_list = []
            for lowcost_index, bounds in lowcost_dict.items():
                lower, upper = bounds
                fi_list.append(lowcost_index)
                lower_list.append(lower)
                upper_list.append(upper)
            attack_model.global_attack(st, ed, 'redundancy', fi_list, lower_list, upper_list, cutoff, confidence_threshold)
    
    return


if __name__=='__main__':
    global args
    args = parse_args()
    main(args)
