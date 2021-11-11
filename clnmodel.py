#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import numpy as np
import random
import torch
import cln
from util import *
from gurobipy import *
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import entropy
import json

GUARD_VAL = 1e-6

def cln_cmp(operator_name, lhs, B, eps):
    if operator_name == '<':
        func = cln.lt
    elif operator_name == '>=':
        func = cln.ge
    elif operator_name == '<=':
        func = cln.le
    elif operator_name == '>':
        func = cln.gt
    elif operator_name == '==':
        func = cln.eq
    return func(lhs, B, eps)

class CLNModel(torch.nn.Module):
    def __init__(self, args_header, args_num_clauses, args_min_atoms, args_max_atoms, \
                args_structure, save_json_path, last_treeid, args_init, args_same, \
                int_indices, nfeat, nlabels, x_train, B=100, eps=0.01, negate=False):
        super(CLNModel, self).__init__()
        self.B = B
        self.eps = eps
        self.args_num_clauses = args_num_clauses
        self.args_min_atoms = args_min_atoms
        self.args_max_atoms = args_max_atoms
        self.args_structure = args_structure
        if self.args_structure != None:
            self.json_content = json.load(open(self.args_structure, 'r'))
        self.save_json_path = save_json_path
        self.last_treeid = last_treeid
        self.args_init = args_init
        self.args_same = args_same
        self.nfeat = nfeat
        self.nlabels = nlabels
        self.x_train = x_train
        # save the best discrete validation tpr
        self.best_val_tpr = 0.0
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.best_model_path = None
        # assume that the min_feat is all 0, max_feat itself represents the range
        if len(x_train) != 0:
            self.max_feat = np.amax(x_train, axis=0)
            self.p80 = np.percentile(x_train, 80, axis=0)
            self.p50 = np.percentile(x_train, 50, axis=0)
            self.std = np.std(x_train, axis=0)
            print("CLNModel.std:", self.std)
        if args_header != None:
            with open(args_header, 'r') as f:
                header = f.readlines()[0].rstrip()
            self.fields = {idx:item for idx, item in enumerate(header.split(','))}
        else:
            self.fields = defaultdict(lambda i: i)
        self.clause_atom_cnt = {} # cid: number of atoms in the clause
        self.clause_atom_samples = {} # cid: list of atom indices from the templates
        self.dead_cids = []
        self.atom_cmp = defaultdict(list) # cid: list of comparisons. <, >=, <=, >, ==
        # cid: list of atoms in cln
        self.cln_clause_atom = {}
        # discrete truth values of the atoms and clauses
        self.atom_states = {}
        self.clause_states = {}
        # for atom that negates another clause.
        self.negate_clause_src = {}
        # for atom tha negates another atom
        self.negate_atom_src = {}
        self.same_atom_src = {}
        # keep track of other cids that refer atoms in this cid
        self.atom_ref = defaultdict(set)
        # store chosen feature indices in the atom.
        self.atom_feat = {}
        # whether some learnable parameters should be positive or negative.
        self.positive_params = {}
        self.negative_params = {}

        self.cln_and_clause = {}
        self.n_leaves = 0
        self.params = torch.nn.ParameterDict()
        self.clause_num = 0
        self.all_cids = []
        self.new_cids = []
        self.del_cids = set([])
        self.parent_cids = set([])
        self.last_cid = -1
        self.cid_label_cnt = {}
        self.cid_entropy = {}
        self.cid_entropy_reduction = {}
        self.cid_parent = {}
        self.int_indices = int_indices
        self.validation_scores = {}
        self.validation_labels = []
        self.cid_acc_gain = {}
        self.cid_loss_gain = {}
        self.cid_info = {}
        # track identical thresholds for the same attribute
        self.check = {}

        population = list(range(self.nfeat))

        # initialize according to the structure of a tree json file
        if self.args_structure != None:
            # read tree rules from model_json
            for treeid, all_paths in enumerate(parse_json(self.json_content)):
                # only initialize treeid from 0 to last_treeid (inclusive)
                if last_treeid != -1 and treeid > last_treeid:
                    break
                #print(treeid, all_paths)
                for all_atoms, leaf_value in all_paths:
                    self.add_one_path(all_atoms, leaf_value)
        else:
            print('args_structure = None')
            raise SystemExit

        #print("self.clause_atom_cnt:", self.clause_atom_cnt)
        #print("self.clause_atom_samples:", self.clause_atom_samples)
        #print("self.atom_cmp:", self.atom_cmp)
        #print("self.negate_clause_src:", self.negate_clause_src)
        return
    
    '''
    Add one path/clause according to parsed json
    '''
    def add_one_path(self, all_atoms, leaf_value):
        cid = self.last_cid + 1
        num_atoms = len(all_atoms)
        self.clause_atom_cnt[cid] = num_atoms
        # each formula contains one path
        samples = []
        aid = 0
        for atom in all_atoms:
            # each atom is a tuple to represent the split
            fid, split_val, yes_no = atom
            # decide whether to make a new atom or refer to an existing one
            if fid in self.int_indices:
                tup_th = math.ceil(split_val)
            else:
                tup_th = split_val
            tup = fid, tup_th, yes_no
            if yes_no == True:
                self.atom_cmp[cid].append('<')
            else:
                self.atom_cmp[cid].append('>=')
            wi = 'w_%s_%s' % (cid, aid)
            ei = 'eta_%s_%s' % (cid, aid)
            num_var = 1
            self.atom_feat[wi] = fid
            if self.args_same == True:
                if tup not in self.check:
                    self.check[tup] = (cid, aid)
                    samples.append(fid)
                    weights = torch.nn.Parameter(torch.ones(size=(num_var,)))
                    if self.args_init == True:
                        eta = torch.nn.Parameter(torch.tensor(float(split_val)))
                        #eta = torch.nn.Parameter(torch.tensor(float(split_val)), requires_grad = False)
                    else:
                        eta = torch.nn.Parameter(torch.tensor(np.random.choice(self.x_train[:, fid])))
                    self.params[wi] = weights
                    self.params[ei] = eta
                else:
                    self.same_atom_src['%s_%s' % (cid, aid)] = self.check[tup]
            else:
                samples.append(fid)
                weights = torch.nn.Parameter(torch.ones(size=(num_var,)))
                if self.args_init == True:
                    eta = torch.nn.Parameter(torch.tensor(float(split_val)))
                    #eta = torch.nn.Parameter(torch.tensor(float(split_val)), requires_grad = False)
                else:
                    eta = torch.nn.Parameter(torch.tensor(np.random.choice(self.x_train[:, fid])))
                self.params[wi] = weights
                self.params[ei] = eta
            aid += 1
        self.clause_atom_samples[cid] = samples

        # one leaf for a clause
        if self.args_init == True:
            leaf = torch.nn.Parameter(torch.tensor(float(leaf_value)))
        else:
            leaf = torch.nn.Parameter(torch.tensor(random.uniform(-1, 1)))
        gate = torch.nn.Parameter(torch.tensor(1.0), requires_grad = False)
        li = 'l_%s' % cid
        gi = 'g_%s' % cid
        self.params[li] = leaf
        self.params[gi] = gate
        self.n_leaves += 1
        self.dead_cids.append(cid)
        self.all_cids.append(cid)
        self.last_cid = cid
        self.clause_num += 1
        self.cid_label_cnt[cid] = [0, 0]
        self.cid_info[cid] = 'init'
        self.cid_entropy[cid] = 1
        self.cid_entropy_reduction[cid] = 0
        self.cid_parent[cid] = -1
        self.args_num_clauses += 1
        self.cid_acc_gain[cid] = 0
        self.cid_loss_gain[cid] = 0
        self.validation_scores[cid] = []
        return
    
    def discrete_logits(self, x, label, label_cnt = False, save_scores = False):
        discrete_clause_atom = {}
        discrete_and_clause = {}
        batch_size = x.shape[0]
        x_atom_states = np.empty([batch_size, 0])
        for cid in self.all_cids:
            discrete_clause_atom[cid] = []
            for aid in range(self.clause_atom_cnt[cid]):
                # check if it's negating another clause
                src_cid = self.negate_clause_src.get('%s_%s' % (cid, aid), None)
                # negate an existing atom
                neg_cid, neg_aid = self.negate_atom_src.get('%s_%s' % (cid, aid), (None, None))
                # same as an existing atom
                same_cid, same_aid = self.same_atom_src.get('%s_%s' % (cid, aid), (None, None))
                if src_cid != None:
                    # NOTE: this is not tested
                    discrete_clause_atom[cid].append(np.invert(np.array([min(stuff) for stuff in discrete_clause_atom[src_cid]])))
                elif neg_aid != None:
                    discrete_clause_atom[cid].append(np.invert((discrete_clause_atom[neg_cid].T)[neg_aid]))
                elif same_aid != None:
                    discrete_clause_atom[cid].append((discrete_clause_atom[same_cid].T)[same_aid])
                else:
                    # get w and eta for the atom
                    wi = 'w_%s_%s' % (cid, aid)
                    ei = 'eta_%s_%s' % (cid, aid)
                    weights = self.params[wi]
                    eta = self.params[ei]
                    # get the chosen x variables
                    cmp_name = self.atom_cmp[cid][aid]
                    chosen = self.atom_feat[wi]
                    xvars = x[:, chosen].view((-1, 1)).cuda()
                    # broadcast multiplication
                    weights = weights.view((-1,1))
                    # (1, 1024)
                    current_atom = compare(cmp_name, \
                        torch.matmul(xvars, weights), eta).cpu().numpy().flatten()
                    discrete_clause_atom[cid].append(current_atom)
            # the empty clause is always True
            if self.clause_atom_cnt[cid] == 0:
                discrete_clause_atom[cid] = np.full(batch_size, True)
            # (1024, atom_num for cid)
            discrete_clause_atom[cid] = np.array(discrete_clause_atom[cid]).T
            #print(cid, discrete_clause_atom[cid].shape)
            x_atom_states = np.append(x_atom_states, discrete_clause_atom[cid], axis=1)
            
            if cid not in self.del_cids:
                # get leaf for the clause
                li = 'l_%s' % cid
                gi = 'g_%s' % cid
                leaf_val = self.params[li].item()
                gate_val = self.params[gi].item()
                discrete_and_clause[cid] = gate_val*leaf_val*np.array([min(stuff) for stuff in discrete_clause_atom[cid]])
                if save_scores == True:
                    self.validation_scores[cid].extend([gate_val*leaf_val*min(stuff) for stuff in discrete_clause_atom[cid]])
                if label_cnt == True:
                    # batch num, e.g., 1024
                    j = 0
                    for all_atoms in discrete_clause_atom[cid]:
                        if min(all_atoms) == True:
                            if label[j] == 0:
                                self.cid_label_cnt[cid][0] += 1
                            else:
                                self.cid_label_cnt[cid][1] += 1
                        j += 1
        if save_scores == True:
            self.validation_labels.extend(label)

        y_pred = sum(list(discrete_and_clause.values()))
        return y_pred, discrete_and_clause, x_atom_states
    
    def discrete_states(self, x, unk_dims):
        # e.g., unk_dims = [0, 1], dimension 0 and 1 have unknown values
        discrete_clause_atom = {}
        # batch size: x.shape[0]
        batch_size = x.shape[0]
        x_atom_states = np.empty([batch_size, 0])
        for cid in self.all_cids:
            discrete_clause_atom[cid] = []
            for aid in range(self.clause_atom_cnt[cid]):
                # check if it's negating another clause
                src_cid = self.negate_clause_src.get('%s_%s' % (cid, aid), None)
                # negate an existing atom
                neg_cid, neg_aid = self.negate_atom_src.get('%s_%s' % (cid, aid), (None, None))
                # same as an existing atom
                same_cid, same_aid = self.same_atom_src.get('%s_%s' % (cid, aid), (None, None))
                if src_cid != None:
                    # NOTE: this is not tested
                    discrete_clause_atom[cid].append(np.invert(np.array([min(stuff) for stuff in discrete_clause_atom[src_cid]])))
                elif neg_aid != None:
                    if (discrete_clause_atom[neg_cid].T)[neg_aid][0] == None:
                        discrete_clause_atom[cid].append((discrete_clause_atom[neg_cid].T)[neg_aid])
                    else:
                        discrete_clause_atom[cid].append(np.invert((discrete_clause_atom[neg_cid].T)[neg_aid]))
                elif same_aid != None:
                    discrete_clause_atom[cid].append((discrete_clause_atom[same_cid].T)[same_aid])
                else:
                    # get w and eta for the atom
                    wi = 'w_%s_%s' % (cid, aid)
                    ei = 'eta_%s_%s' % (cid, aid)
                    weights = self.params[wi]
                    eta = self.params[ei]
                    # get the chosen x variables
                    cmp_name = self.atom_cmp[cid][aid]
                    chosen = self.atom_feat[wi]
                    if chosen in unk_dims:
                        discrete_clause_atom[cid].append(np.full(batch_size, None))
                    else:
                        xvars = x[:, chosen].view((-1, 1)).cuda()
                        # broadcast multiplication
                        weights = weights.view((-1,1))
                        # (1, 1024)
                        current_atom = compare(cmp_name, \
                            torch.matmul(xvars, weights), eta).cpu().numpy().flatten()
                        discrete_clause_atom[cid].append(current_atom)
            
            # the empty clause is always True
            if self.clause_atom_cnt[cid] == 0:
                discrete_clause_atom[cid] = np.full(batch_size, True)
            # (1024, atom_num for cid)
            discrete_clause_atom[cid] = np.array(discrete_clause_atom[cid]).T
            x_atom_states = np.append(x_atom_states, discrete_clause_atom[cid], axis=1)
            
        return x_atom_states
    
    def save_as_json(self):
        tree_model = self.json_content['learner']['gradient_booster']['model']
        tree_model['gbtree_model_param']['num_trees'] = str(self.clause_num)
        # empty the existing "trees"
        tree_model['trees'] = []
        tree_model['tree_info'] = [0]*self.clause_num
        # write each clause as a tree
        pp_clause = {}
        pp_atoms = defaultdict(list)
        split_threshold = defaultdict(list)
        split_indices = defaultdict(list)
        next_left_child = defaultdict(list)
        for cid in self.all_cids:
            for aid in range(self.clause_atom_cnt[cid]):
                # check if it's negating another clause
                src_cid = self.negate_clause_src.get('%s_%s' % (cid, aid), None)
                # negate an existing atom
                neg_cid, neg_aid = self.negate_atom_src.get('%s_%s' % (cid, aid), (None, None))
                # same as an existing atom
                same_cid, same_aid = self.same_atom_src.get('%s_%s' % (cid, aid), (None, None))
                if src_cid != None:
                    pp_atoms[cid].append(('not (%s)' % pp_clause[src_cid], '', ''))
                elif neg_aid != None:
                    src_atom = pp_atoms[neg_cid][neg_aid]
                    pp_atoms[cid].append(('not (%s %s %s)' % (src_atom[0], src_atom[1], src_atom[2]), '', ''))
                    split_threshold[cid].append(split_threshold[neg_cid][neg_aid])
                    split_indices[cid].append(chosen)
                    next_left_child[cid].append(not next_left_child[neg_cid][neg_aid])
                elif same_aid != None:
                    try:
                        src_atom = pp_atoms[same_cid][same_aid]
                        split_threshold[cid].append(split_threshold[same_cid][same_aid])
                        split_indices[cid].append(chosen)
                        next_left_child[cid].append(next_left_child[same_cid][same_aid])
                    except IndexError:
                        print('cid', cid, 'aid', aid)
                        print('pp_atoms[same_cid][same_aid]', same_cid, same_aid)
                    pp_atoms[cid].append(src_atom)
                else:
                    # get w and eta for the atom
                    wi = 'w_%s_%s' % (cid, aid)
                    ei = 'eta_%s_%s' % (cid, aid)
                    weights = self.params[wi].data.cpu().numpy()
                    eta = self.params[ei].item()

                    # get the chosen x variables
                    cmp_name = self.atom_cmp[cid][aid]
                    chosen = self.atom_feat[wi]
                    lhs_terms = '%s*%s' % (weights[0], self.fields[chosen])
                    pp_atoms[cid].append((lhs_terms, cmp_name, eta))
                    weight = weights[0]

                    if weight != 0:
                        if weight > 0 and cmp_name == '<':
                            threshold = eta / weight
                            ### p_negate = False
                            next_left_child[cid].append(True)
                        elif weight < 0 and cmp_name == '>=':
                            threshold = eta / weight + GUARD_VAL
                            ### p_negate = False
                            next_left_child[cid].append(True)
                        elif weight > 0 and cmp_name == '>=':
                            threshold = eta / weight
                            ### p_negate = True
                            next_left_child[cid].append(False)
                        else:
                            # weight < 0 and cmp_name == '<'
                            threshold = eta / weight + GUARD_VAL
                            #p_negate = True
                            next_left_child[cid].append(False)
                    else:
                        # use np.inf and -np.inf to track threshold
                        if (eta >= 0 and cmp_name == '<') or (eta == 0 and cmp_name == '>=') or (eta < 0 and cmp_name == '>='):
                            threshold = np.inf
                        else:
                            threshold = -np.inf
                        next_left_child[cid].append(True)
                    split_threshold[cid].append(threshold)
                    split_indices[cid].append(chosen)
            #print(cid, split_threshold[cid])
            #print(cid, split_indices[cid])
            #print(cid, next_left_child[cid])

            if cid not in self.del_cids:
                # len(pp_atoms[cid]) is the number of nodes for this tree
                # initialize arrays
                num_atoms = self.clause_atom_cnt[cid]
                num_nodes = 2 * num_atoms + 1
                cur_tree = {"categories": [], "categories_nodes": [], \
                        "categories_segments": [], "categories_sizes": [], \
                        "id": cid, \
                        "tree_param": {
                          "num_deleted": "0",
                          "num_feature": "15",
                          "num_nodes": str(num_nodes),
                          "size_leaf_vector": "0"
                        }
                 }
                cur_tree['base_weights'] = [0.0]*num_nodes
                cur_tree['default_left'] = [True]*num_nodes
                cur_tree['left_children'] = [-1]*num_nodes
                cur_tree['loss_changes'] = [0.0]*num_nodes
                cur_tree['parents'] = [-1]*num_nodes
                cur_tree['right_children'] = [-1]*num_nodes
                cur_tree['split_conditions'] = [0.0]*num_nodes
                cur_tree['split_indices'] = [0]*num_nodes
                cur_tree['split_type'] = [0]*num_nodes
                cur_tree['sum_hessian'] = [0.0]*num_nodes
                
                parent_id = MAX_INT32
                for j in range(num_atoms + 1):
                    #print('parent_id', parent_id)
                    # root node
                    if j == 0:
                        cur_tree['parents'][0] = parent_id
                        cur_node_idx = 0
                    # non root node
                    else:
                        if j == 1:
                            parent_left = 1
                            parent_right = 2
                        else:
                            parent_left = 2*j-1
                            parent_right = 2*j
                        if next_left_child[cid][j-1] == True:
                            # current is the left child of the parent
                            cur_node_idx = parent_left
                            sibling = parent_right
                            cur_tree['left_children'][parent_id] = cur_node_idx
                            cur_tree['right_children'][parent_id] = sibling
                        else:
                            # current is the right child of the parent
                            cur_node_idx = parent_right
                            sibling = parent_left
                            cur_tree['default_left'][parent_id] = False
                            cur_tree['right_children'][parent_id] = cur_node_idx
                            cur_tree['left_children'][parent_id] = sibling
                        cur_tree['parents'][cur_node_idx] = parent_id
                        cur_tree['parents'][sibling] = parent_id
                        cur_tree['left_children'][sibling] = -1
                        cur_tree['right_children'][sibling] = -1
                    
                    # non leaf node, add split info
                    if j != num_atoms:
                        cur_tree['split_conditions'][cur_node_idx] = split_threshold[cid][j]
                        cur_tree['split_indices'][cur_node_idx] = split_indices[cid][j]
                    # leaf node
                    else:
                        # get leaf for the clause
                        li = 'l_%s' % cid
                        leaf_val = self.params[li].item()
                        cur_tree['base_weights'][cur_node_idx] = leaf_val
                        cur_tree['split_conditions'][cur_node_idx] = leaf_val 
                        cur_tree['left_children'][cur_node_idx] = -1
                        cur_tree['right_children'][cur_node_idx] = -1
                    parent_id = cur_node_idx

            tree_model['trees'].append(cur_tree)
        json.dump(self.json_content, open(self.save_json_path, 'w'))
        return

    def discrete_pred(self, x, label, label_cnt = False, save_scores = False):
        y_pred, _, _ = self.discrete_logits(x, label, label_cnt, save_scores)
        s = lambda x: 1/(1 + np.exp(-x))
        return s(y_pred).flatten().T

    def discrete_clause_truth(self, x):
        discrete_clause_atom = {}
        clause_truth = {}
        for cid in self.all_cids:
            discrete_clause_atom[cid] = []
            for aid in range(self.clause_atom_cnt[cid]):
                # check if it's negating another clause
                src_cid = self.negate_clause_src.get('%s_%s' % (cid, aid), None)
                # negate an existing atom
                neg_cid, neg_aid = self.negate_atom_src.get('%s_%s' % (cid, aid), (None, None))
                # same as an existing atom
                same_cid, same_aid = self.same_atom_src.get('%s_%s' % (cid, aid), (None, None))
                if src_cid != None:
                    # NOTE: this is not tested
                    discrete_clause_atom[cid].append(np.invert(np.array([min(stuff) for stuff in discrete_clause_atom[src_cid]])))
                elif neg_aid != None:
                    discrete_clause_atom[cid].append(np.invert((discrete_clause_atom[neg_cid].T)[neg_aid]))
                elif same_aid != None:
                    discrete_clause_atom[cid].append((discrete_clause_atom[same_cid].T)[same_aid])
                else:
                    # get w and eta for the atom
                    wi = 'w_%s_%s' % (cid, aid)
                    ei = 'eta_%s_%s' % (cid, aid)
                    weights = self.params[wi]
                    eta = self.params[ei]
                    # get the chosen x variables
                    cmp_name = self.atom_cmp[cid][aid]
                    chosen = self.atom_feat[wi]
                    xvars = x[:, chosen].view((-1,1)).cuda()
                    # broadcast multiplication
                    weights = weights.view((-1,1))
                    # (1, 1024)
                    current_atom = compare(cmp_name, \
                        torch.matmul(xvars, weights), eta).cpu().numpy().flatten()
                    discrete_clause_atom[cid].append(current_atom)

            # (1024, atom_num for cid)
            discrete_clause_atom[cid] = np.array(discrete_clause_atom[cid]).T

            if cid not in self.del_cids:
                clause_truth[cid] = np.array([min(stuff) for stuff in discrete_clause_atom[cid]])
        return clause_truth

    def save_numpy_scores(self):
        for cid in self.all_cids:
            if cid in self.del_cids:
                continue
            self.validation_scores[cid] = np.array([self.validation_scores[cid]])

    def subset_pred(self, cid_set):
        newdict = {cid: self.validation_scores[cid] for cid in cid_set}
        y_pred = sum(list(newdict.values()))
        s = lambda x: 1/(1 + np.exp(-x))
        return s(y_pred).flatten().T

    def subset_acc(self, cid_set):
        scores = self.subset_pred(cid_set)
        if self.nlabels == 2:
            preds = (scores>=0.5).astype(float)
        else:
            preds = scores.max(1)[1]
        num_correct = (preds == self.validation_labels).sum()
        num_samples = len(self.validation_labels)
        acc = float(num_correct) / num_samples
        return acc

    def compute_acc_gain(self):
        # compute the acc gain by each individual clause in the model
        active_cids = set(self.all_cids) - set(self.del_cids)
        all_acc = self.subset_acc(active_cids)
        for cid in active_cids:
            # make a subset all_cids - del_cids - cid
            cid_set = active_cids - set([cid])
            acc = self.subset_acc(cid_set)
            gain = all_acc - acc
            self.cid_acc_gain[cid] = gain
        return

    def subset_loss(self, cid_set):
        scores = self.subset_pred(cid_set)
        num_samples = len(self.validation_labels)
        loss = -(1.0/num_samples) * (np.dot(np.log(scores), np.array(self.validation_labels).T) + np.dot(np.log(1-scores), (1-np.array(self.validation_labels)).T))
        return loss

    def compute_loss_gain(self):
        # compute the loss gain by each individual clause in the model
        active_cids = set(self.all_cids) - set(self.del_cids)
        all_loss = self.subset_loss(active_cids)
        for cid in active_cids:
            # make a subset all_cids - del_cids - cid
            cid_set = active_cids - set([cid])
            loss = self.subset_loss(cid_set)
            gain = loss - all_loss
            self.cid_loss_gain[cid] = gain
        return

    def count_classes(self, x_train, y_train, batch_size):
        cid_label_counts = {}
        x_train_tensor = torch.from_numpy(x_train).float()
        y_train_tensor = torch.from_numpy(y_train).float()
        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        for x_batch, y_batch in train_loader:
            x_batch = x_batch
            y_batch = y_batch
            clause_truth = self.discrete_clause_truth(x_batch)

            # count y_batch classes for each activated clause
            #print(clause_truth[self.last_cid].size)
            #print(clause_truth)

            # among the True data points, count positive and negative labels
            for cid, truth_list in clause_truth.items():
                true_batch = y_batch[truth_list == True]
                pos_sum = int(sum(true_batch))
                neg_sum = len(true_batch) - pos_sum
                try:
                    cid_label_counts[cid][0] += neg_sum
                    cid_label_counts[cid][1] += pos_sum
                except KeyError:
                    cid_label_counts[cid] = [neg_sum, pos_sum]
                #print(cid, pos_sum, neg_sum)

        return cid_label_counts

    def save_current_clauses(self):
        for i, val in self.params.items():
            val.requires_grad = False
        return

    def add_constraint(self, projector, changed_atoms, pred_scores, prop_name, *args):
        '''
        assume that len(changed_atoms) <= num_atoms for this clause
        '''
        cid = self.last_cid+1
        aid = 0
        before_score = 0
        after_score = 0
        if len(changed_atoms) == 0:
            return 0

        addnum = 0
        before_true_cids = set()
        after_true_cids = set()
        for ref_cid, all_scores in pred_scores[1].items():
            # the score before is not the same as the score after for ref_cid
            if pred_scores[0][ref_cid][0] != pred_scores[1][ref_cid][0]:
                if pred_scores[0][ref_cid][0] != 0 and pred_scores[0][ref_cid][0] != None:
                    before_score += pred_scores[0][ref_cid][0]
                    before_true_cids.add(ref_cid)
                # all_scores = pred_scores[1][ref_cid]
                if all_scores[0] != 0 and all_scores[0] != None:
                    after_score += all_scores[0]
                    after_true_cids.add(ref_cid)
        print('before_score', before_score, 'after_score', after_score)
        print('before_true_cids', before_true_cids)
        print('after_true_cids', after_true_cids)
        
        '''
        Add constraints
        '''
        if prop_name == 'monotonicity':
            if before_score > after_score:
                # the new after_score >= new before_score
                print('\n***** adding constraint: sum(%s) <= sum(%s)' % \
                        (['l_%s' % ref_cid for ref_cid in before_true_cids], \
                         ['l_%s' % ref_cid for ref_cid in after_true_cids]))
                projector.add_constr_sum_le_sum(before_true_cids, after_true_cids)
                addnum += 1
            else:
                # the new after_score <= new before_score
                print('\n***** adding constraint: sum(%s) <= sum(%s)' % \
                        (['l_%s' % ref_cid for ref_cid in after_true_cids], \
                         ['l_%s' % ref_cid for ref_cid in before_true_cids]))
                projector.add_constr_sum_le_sum(after_true_cids, before_true_cids)
                addnum += 1

        elif prop_name == 'stability' or prop_name == 'eps':
            output_diff = args[0]
            # | new activation value - all before true activation values | <= output_diff
            print('\n***** adding constraint: | sum(%s) - sum(%s) | <= %s' % ( \
                    ['l_%s' % ref_cid for ref_cid in before_true_cids], \
                    ['l_%s' % ref_cid for ref_cid in after_true_cids], \
                    output_diff))
            projector.add_constr_absdiff_le_c(before_true_cids, after_true_cids, output_diff)
            addnum += 1
        elif prop_name == 'lowcost' or prop_name == 'redundancy':
            cutoff = args[0]
            confidence = args[1]
            cname = args[2]
            # the difference of before - after should be smaller than inv_sigmoid(confidence) - inv_sigmoid(cutoff)
            diff_bound = inv_sigmoid(confidence)-inv_sigmoid(cutoff) + GUARD_VAL
            print('\n***** adding constraint: sum(%s) - sum(%s) <= inv_sigmoid(%f) - inv_sigmoid(%f) + GUARD_VAL = %f' % ( \
                    ['l_%s' % ref_cid for ref_cid in before_true_cids], \
                    ['l_%s' % ref_cid for ref_cid in after_true_cids], \
                    confidence, cutoff, diff_bound))
            projector.add_constr_diff_le_c(before_true_cids, after_true_cids, diff_bound, cname)
            addnum += 1
        elif prop_name == 'lowcost_nond':
            # the new after_score >= new before_score
            print('\n***** adding constraint: sum(%s) <= sum(%s)' % \
                    (['l_%s' % ref_cid for ref_cid in before_true_cids], \
                        ['l_%s' % ref_cid for ref_cid in after_true_cids]))
            projector.add_constr_sum_le_sum(before_true_cids, after_true_cids)
            addnum += 1
        return addnum
    
    def update_lowcost_rhs(self, projector, cutoff, confidence, total_rounds, cname):
        # find constraint
        cstr = projector.grb.getConstrByName(cname)
        # recompute RHS
        cstr.RHS = (inv_sigmoid(confidence)-inv_sigmoid(cutoff))/float(total_rounds)+GUARD_VAL
        projector.grb.update()
        return

    def remove_cid(self, cid, projector):
        print('remove old Clause ID', cid, ':', self.cid_info[cid])
        self.del_cids.add(cid)

        li = 'l_%s' % cid
        gi = 'g_%s' % cid
        self.params[li].requires_grad = False
        self.params[gi].requires_grad = False

        # try:
        #     self.all_cids.remove(cid)
        # except ValueError:
        #     pass
        if cid in self.new_cids:
            self.new_cids.remove(cid)
        try:
            del self.cln_clause_atom[cid]
        except KeyError:
            pass
        try:
            del self.cln_and_clause[cid]
        except KeyError:
            pass
        try:
            del self.atom_states[cid]
        except KeyError:
            pass

        # set the leaf variable to be zero in the projector model
        if projector != None:
            projector.grb.addConstr(projector.L[cid] == 0)

        self.n_leaves -= 1
        self.clause_num -= 1

        return

    def compute_entropy(self):
        for cid, label_counts in self.cid_label_cnt.items():
            total = float(sum(label_counts))
            if total != 0:
                ent = entropy([label_counts[0]/total, label_counts[1]/total])
                self.cid_entropy[cid] = ent
        return

    def compute_entropy_reduction(self):
        for cid, ent in self.cid_entropy.items():
            parent_cid = self.cid_parent[cid]
            if parent_cid == -1:
                parent_entropy = 1
            else:
                parent_entropy = self.cid_entropy[parent_cid]
            self.cid_entropy_reduction[cid] = parent_entropy - ent
        return

    def compute_information_gain(self, y_validation):
        for cid, ent in self.cid_entropy.items():
            parent_cid = self.cid_parent[cid]
            if parent_cid == -1:
                parent_positive = sum(y_validation)
                parent_negative = len(y_validation) - sum(y_validation)
                parent_entropy = entropy([parent_positive, parent_negative])
            else:
                parent_entropy = self.cid_entropy[parent_cid]
                parent_positive = self.cid_label_cnt[parent_cid][1]
                parent_negative = self.cid_label_cnt[parent_cid][0]
            parent_total = parent_positive + parent_negative
            cur_true_total = self.cid_label_cnt[cid][0] + self.cid_label_cnt[cid][1]
            cur_true_ent = ent
            #cur_false_total = parent_total - cur_true_total
            cur_false_positive = parent_positive - self.cid_label_cnt[cid][1]
            cur_false_negative = parent_negative - self.cid_label_cnt[cid][0]
            cur_false_total = cur_false_positive + cur_false_negative
            if cur_false_total == 0:
                cur_false_ent = 0
            else:
                cur_false_ent = entropy([cur_false_positive, cur_false_negative])
            if parent_total == 0 or cur_false_total == 0:
                self.cid_entropy_reduction[cid] = 0
            else:
                self.cid_entropy_reduction[cid] = parent_entropy - cur_true_total/float(parent_total) * cur_true_ent - cur_false_total/float(parent_total) * cur_false_ent
        return

    def prettyprint(self, print_cid = None):
        pp_clause = {}
        pp_atoms = defaultdict(list)
        for cid in self.all_cids:
            for aid in range(self.clause_atom_cnt[cid]):
                # check if it's negating another clause
                src_cid = self.negate_clause_src.get('%s_%s' % (cid, aid), None)
                # negate an existing atom
                neg_cid, neg_aid = self.negate_atom_src.get('%s_%s' % (cid, aid), (None, None))
                # same as an existing atom
                same_cid, same_aid = self.same_atom_src.get('%s_%s' % (cid, aid), (None, None))
                if src_cid != None:
                    pp_atoms[cid].append(('not (%s)' % pp_clause[src_cid], '', ''))
                elif neg_aid != None:
                    src_atom = pp_atoms[neg_cid][neg_aid]
                    pp_atoms[cid].append(('not (%s %s %s)' % (src_atom[0], src_atom[1], src_atom[2]), '', ''))
                elif same_aid != None:
                    try:
                        src_atom = pp_atoms[same_cid][same_aid]
                    except IndexError:
                        print('cid', cid, 'aid', aid)
                        print('pp_atoms[same_cid][same_aid]', same_cid, same_aid)
                    pp_atoms[cid].append(src_atom)
                else:
                    # get w and eta for the atom
                    wi = 'w_%s_%s' % (cid, aid)
                    ei = 'eta_%s_%s' % (cid, aid)
                    weights = self.params[wi].data.cpu().numpy()
                    eta = self.params[ei].item()

                    # get the chosen x variables
                    cmp_name = self.atom_cmp[cid][aid]
                    chosen = self.atom_feat[wi]
                    lhs_terms = '%s*%s' % (weights[0], self.fields[chosen])
                    pp_atoms[cid].append((lhs_terms, cmp_name, eta))

            if cid not in self.del_cids:
                path_str = '\n\t\t\t\t and '.join(['(%s %s %s)' % (atom[0], atom[1], atom[2]) \
                            for j, atom in enumerate(pp_atoms[cid])])
                # get leaf for the clause
                li = 'l_%s' % cid
                leaf_val = self.params[li].item()
                gi = 'g_%s' % cid
                gate_val = self.params[gi].item()
                if leaf_val > 0:
                    sign = '++'
                else:
                    sign = '--'
                if print_cid != None and cid != print_cid:
                    pass
                else:
                    print("   Label Cnt: %s" % self.cid_label_cnt[cid])
                    print("   Entropy: %s" % self.cid_entropy[cid])
                    print("   Information Gain: %s" % self.cid_entropy_reduction[cid])
                    print("   Acc Gain: %s" % self.cid_acc_gain[cid])
                    print("   Loss Gain: %s" % self.cid_loss_gain[cid])
                    print("   Gate val: %s" % gate_val)
                    print("{} Activation val: {}; Clause ID: {}; Formula: \n\t\t\t\t{}\n".format(\
                            sign, leaf_val, cid, path_str))
                pp_clause[cid] = path_str
        return

    def annotate(self, strong_set, weak_set_1, weak_set_2):
        pp_clause = {}
        pp_atoms = defaultdict(list)
        for cid in self.all_cids:
            for aid in range(self.clause_atom_cnt[cid]):
                # check if it's negating another clause
                src_cid = self.negate_clause_src.get('%s_%s' % (cid, aid), None)
                # negate an existing atom
                neg_cid, neg_aid = self.negate_atom_src.get('%s_%s' % (cid, aid), (None, None))
                # same as an existing atom
                same_cid, same_aid = self.same_atom_src.get('%s_%s' % (cid, aid), (None, None))
                if src_cid != None:
                    pp_atoms[cid].append(('not (%s)' % pp_clause[src_cid], '', ''))
                elif neg_aid != None:
                    src_atom = pp_atoms[neg_cid][neg_aid]
                    pp_atoms[cid].append(('not (%s %s %s %s)' % (src_atom[0], src_atom[1], src_atom[2], src_atom[3]), '', '', ''))
                elif same_aid != None:
                    src_atom = pp_atoms[same_cid][same_aid]
                    pp_atoms[cid].append(src_atom)
                else:
                    # get w and eta for the atom
                    wi = 'w_%s_%s' % (cid, aid)
                    ei = 'eta_%s_%s' % (cid, aid)
                    weights = self.params[wi].data.cpu().numpy()
                    eta = self.params[ei].item()

                    # get the chosen x variables
                    cmp_name = self.atom_cmp[cid][aid]
                    chosen = self.atom_feat[wi]
                    if chosen in strong_set:
                        atom_label = '***STRONG***'
                    elif chosen in weak_set_1:
                        atom_label = '__w_1__'
                    elif chosen in weak_set_2:
                        atom_label = '__w_2__'
                    lh_terms = '%s*%s' % (weights[0], self.fields[chosen])
                    pp_atoms[cid].append((lhs_terms, cmp_name, eta, atom_label))

            if cid not in self.del_cids:
                path_str = '\n\t\t\t\t and '.join(['(%s %s %s %s)' % (atom[0], atom[1], atom[2], atom[3]) \
                            for j, atom in enumerate(pp_atoms[cid])])
                # get leaf for the clause
                li = 'l_%s' % cid
                leaf_val = self.params[li].item()
                if leaf_val > 0:
                    sign = '++'
                else:
                    sign = '--'
                print("{} Activation val: {}; Clause ID: {}; Formula: \n\t\t\t\t{}\n".format(\
                        sign, leaf_val, cid, path_str))
                pp_clause[cid] = path_str
        return

    def forward(self, x, label, train_cnt = False):
        if self.nlabels ==2:
            return self.binary_forward(x, label, train_cnt)
        else:
            return self.multiclass_forward(x, label)

    def reset_label_cnt(self):
        for cid in self.cid_label_cnt.keys():
            if cid not in self.del_cids:
                self.cid_label_cnt[cid] = [0, 0]
                self.cid_entropy[cid] = 1
                self.cid_acc_gain[cid] = 0
                self.cid_loss_gain[cid] = 0
                self.validation_scores[cid] = []
        self.validation_labels = []
        return

    def binary_forward(self, x, label, train_cnt):
        eps = self.eps

        for cid in self.all_cids:
            self.cln_clause_atom[cid] = []
            self.atom_states[cid] = []
            # compute cln_clause_atom[cid] for all cids
            for aid in range(self.clause_atom_cnt[cid]):
                # check if it's negating another clause
                src_cid = self.negate_clause_src.get('%s_%s' % (cid, aid), None)
                # negate an existing atom
                neg_cid, neg_aid = self.negate_atom_src.get('%s_%s' % (cid, aid), (None, None))
                # same as an existing atom
                same_cid, same_aid = self.same_atom_src.get('%s_%s' % (cid, aid), (None, None))
                if src_cid != None:
                    self.cln_clause_atom[cid].append(cln.neg(cln.prod_tnorm(self.cln_clause_atom[src_cid])))
                    self.atom_states[cid].append(np.invert(self.clause_states[src_cid]))
                elif neg_aid != None:
                    self.cln_clause_atom[cid].append(cln.neg(self.cln_clause_atom[neg_cid][neg_aid]))
                    self.atom_states[cid].append(np.invert(self.atom_states[neg_cid][neg_aid]))
                elif same_aid != None:
                    self.cln_clause_atom[cid].append(self.cln_clause_atom[same_cid][same_aid])
                    self.atom_states[cid].append(self.atom_states[same_cid][same_aid])
                else:
                    # get w and eta for the atom
                    wi = 'w_%s_%s' % (cid, aid)
                    ei = 'eta_%s_%s' % (cid, aid)
                    weights = self.params[wi]
                    eta = self.params[ei]

                    # get the chosen x variables
                    cmp_name = self.atom_cmp[cid][aid]
                    chosen = self.atom_feat[wi]

                    model_weights = (weights.data.cpu().numpy() * (1000000)).astype(int)
                    model_eta = int(eta * (1000000))
                    vars = [x[:, chosen]]
                    lhs = sum([model_weights[i]*vars[i] for i in range(len(vars))])
                    self.atom_states[cid].append(compare(cmp_name, lhs, model_eta).cpu().numpy())
                    xvars = x[:, chosen].view((-1, 1)).cuda()
                    # broadcast multiplication
                    weights = weights.view((-1,1))
                    self.cln_clause_atom[cid].append(cln_cmp(cmp_name, \
                        torch.matmul(xvars, weights) - eta, self.B/self.std[chosen], eps))
            if self.clause_atom_cnt[cid] == 0:
                self.atom_states[cid] = np.full(x.shape[0], True)
            self.atom_states[cid] = np.array(self.atom_states[cid])
            # compute cln_and_clause[cid] for clauses that are not deleted
            if cid not in self.del_cids:
                # get leaf for the clause
                li = 'l_%s' % cid
                gi = 'g_%s' % cid
                leaf_val = self.params[li]
                gate_val = self.params[gi]

                self.cln_and_clause[cid] = gate_val*leaf_val*cln.prod_tnorm(self.cln_clause_atom[cid])
                # go over each column in the self.atom_states[cid]
                j = 0
                for x_atoms in self.atom_states[cid].T:
                    if False in x_atoms:
                        self.clause_states[cid] = False
                    else:
                        self.clause_states[cid] = True
                        try:
                            self.dead_cids.remove(cid)
                        except ValueError:
                            pass
                        # count the number of positive and negative labels
                        # if self.cid_label_cnt.get(cid, None) == None:
                        #     self.cid_label_cnt[cid] = [0, 0]
                        if train_cnt == True:
                            if label[j] == 0:
                                self.cid_label_cnt[cid][0] += 1
                            else:
                                self.cid_label_cnt[cid][1] += 1
                    j += 1
        y_pred = sum(list(self.cln_and_clause.values()))
        #y_pred = cln.prod_tconorm(list(self.cln_and_clause.values()))
        # y_pred = torch.tensor([1-torch.sigmoid(pred_out), torch.sigmoid(pred_out)], requires_grad=True)
        s = torch.nn.Sigmoid()
        return s(y_pred).flatten().T

    def multiclass_forward(self, x, label):
        return None
