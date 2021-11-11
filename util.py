#!/usr/bin/env python
# coding: utf-8

import operator
import math
import numpy as np

MAX_INT32 = pow(2, 31) - 1

def sigmoid(val):
    return 1/(1 + np.exp(-val))

def inv_sigmoid(val):
    return math.log(val/(1-val))

def f1_score(precision, recall):
    return 2*precision*recall/float(precision+recall)

def compare(operator_name, a, b):
    if operator_name == '<':
        func = operator.lt
    elif operator_name == '>=':
        func = operator.ge
    elif operator_name == '<=':
        func = operator.le
    elif operator_name == '>':
        func = operator.gt
    elif operator_name == '==':
        func = operator.eq
    return func(a, b)

def py_and_list(input):
    res = input[0]
    for i in range(1, len(input)):
        res = res and input[i]
    return bool(res)

def shuffle_data(x, y):
    idx = np.arange(0 , len(x))
    np.random.shuffle(idx)
    x_shuffle = np.array([x[i] for i in idx])
    y_shuffle = np.array([y[i] for i in idx])
    return x_shuffle, y_shuffle

def dfs(node_id, tree_json, parsed_tree, cur_path):
    # add current node
    left_child = tree_json['left_children'][node_id]
    right_child = tree_json['right_children'][node_id]
    # return if it is a leaf
    if left_child == -1 and right_child == -1:
        leaf_value = tree_json['base_weights'][node_id]
        # activation value could be learned as zero for CLN
        if leaf_value != 0:
            parsed_tree.append((cur_path, leaf_value))
    else:
        # otherwise recurse
        if left_child != -1:
            fid = tree_json['split_indices'][node_id]
            split_val = tree_json['split_conditions'][node_id]
            left_path = cur_path.copy()
            left_path.append((fid, split_val, True))
            dfs(left_child, tree_json, parsed_tree, left_path)
        if right_child != -1:
            fid = tree_json['split_indices'][node_id]
            split_val = tree_json['split_conditions'][node_id]
            right_path = cur_path.copy()
            right_path.append((fid, split_val, False))
            dfs(right_child, tree_json, parsed_tree, right_path)
    return

def parse_json(json_content):
    tree_model = json_content['learner']['gradient_booster']['model']
    num_trees = int(tree_model['gbtree_model_param']['num_trees'])
    all_trees = tree_model['trees']
    all_parsed_trees = []
    for tree_id in range(num_trees):
        parsed_tree = []
        tree_json = all_trees[tree_id]
        if tree_json['parents'][0] == MAX_INT32:
            root_id = 0
            dfs(root_id, tree_json, parsed_tree, [])
            if parsed_tree == []:
                parsed_tree = [([], 0.0)]
            all_parsed_trees.append(parsed_tree)
        else:
            print('Exception: root node is not 0')
            raise SystemExit
    return all_parsed_trees

def check_states(cln_model, start_cid, end_cid, int_indices, test_input):
    x = test_input
    sum_func = sum
    and_func = py_and_list
    not_func = operator.not_

    smt_clause_atom = {}
    smt_and_clause = {}
    leaf_ret = {}

    for cid in range(start_cid, end_cid + 1):
        smt_clause_atom[cid] = []
        for aid in range(cln_model.clause_atom_cnt[cid]):
            # check if it's negating another clause
            src_cid = cln_model.negate_clause_src.get('%s_%s' % (cid, aid), None)
            # negate an existing atom
            neg_cid, neg_aid = cln_model.negate_atom_src.get('%s_%s' % (cid, aid), (None, None))
            # same as an existing atom
            same_cid, same_aid = cln_model.same_atom_src.get('%s_%s' % (cid, aid), (None, None))
            if src_cid != None:
                # if one of the atoms in the conjunction is false
                # then the negation of the conjunction is true
                # else the negation is false
                if False in smt_clause_atom[src_cid]:
                    smt_clause_atom[cid].append(True)
                elif None in smt_clause_atom[src_cid]:
                    smt_clause_atom[cid].append(None)
                else:
                    smt_clause_atom[cid].append(False)
            elif neg_aid != None:
                src_atom = smt_clause_atom[neg_cid][neg_aid]
                if src_atom == None:
                    smt_clause_atom[cid].append(None)
                else:
                    smt_clause_atom[cid].append(not src_atom)
            elif same_aid != None:
                src_atom = smt_clause_atom[same_cid][same_aid]
                smt_clause_atom[cid].append(src_atom)
            else:
                # get w and eta for the atom
                wi = 'w_%s_%s' % (cid, aid)
                ei = 'eta_%s_%s' % (cid, aid)
                weights = cln_model.params[wi].data.cpu().numpy()
                eta = cln_model.params[ei].item()

                # get the chosen x variables
                cmp_name = cln_model.atom_cmp[cid][aid]
                chosen = cln_model.atom_feat[wi]
                xvars = [x[chosen]]
                if None in xvars:
                    smt_clause_atom[cid].append(None)
                else:
                    lhs = sum_func([weights[i]*xvars[i] for i in range(len(xvars))])
                    smt_clause_atom[cid].append( \
                        compare(cmp_name, lhs, eta))

        if cid not in cln_model.del_cids:
            # get leaf for the clause
            li = 'l_%s' % cid
            leaf_val = cln_model.params[li].item()

            if leaf_val > 0:
                max_leaf_val = leaf_val
                min_leaf_val = 0
            else:
                max_leaf_val = 0
                min_leaf_val = leaf_val

            if False in smt_clause_atom[cid]:
                smt_and_clause[cid] = False
                leaf_ret[cid] = [0, 0, 0]
            elif None in smt_clause_atom[cid]:
                smt_and_clause[cid] = None
                leaf_ret[cid] = [None, min_leaf_val, max_leaf_val]
            else:
                smt_and_clause[cid] = True
                leaf_ret[cid] = [leaf_val, leaf_val, leaf_val]
    return smt_clause_atom, smt_and_clause, leaf_ret

def check_states_select(cln_model, cid_list, int_indices, test_input):
    x = test_input
    sum_func = sum
    and_func = py_and_list
    not_func = operator.not_

    smt_clause_atom = {}
    smt_and_clause = {}
    leaf_ret = {}

    for cid in cid_list:
        smt_clause_atom[cid] = []
        for aid in range(cln_model.clause_atom_cnt[cid]):
            # check if it's negating another clause
            src_cid = cln_model.negate_clause_src.get('%s_%s' % (cid, aid), None)
            # negate an existing atom
            neg_cid, neg_aid = cln_model.negate_atom_src.get('%s_%s' % (cid, aid), (None, None))
            # same as an existing atom
            same_cid, same_aid = cln_model.same_atom_src.get('%s_%s' % (cid, aid), (None, None))
            if src_cid != None:
                # if one of the atoms in the conjunction is false
                # then the negation of the conjunction is true
                # else the negation is false
                if False in smt_clause_atom[src_cid]:
                    smt_clause_atom[cid].append(True)
                elif None in smt_clause_atom[src_cid]:
                    smt_clause_atom[cid].append(None)
                else:
                    smt_clause_atom[cid].append(False)
            elif neg_aid != None:
                src_atom = smt_clause_atom[neg_cid][neg_aid]
                if src_atom == None:
                    smt_clause_atom[cid].append(None)
                else:
                    smt_clause_atom[cid].append(not src_atom)
            elif same_aid != None:
                src_atom = smt_clause_atom[same_cid][same_aid]
                smt_clause_atom[cid].append(src_atom)
            else:
                # get w and eta for the atom
                wi = 'w_%s_%s' % (cid, aid)
                ei = 'eta_%s_%s' % (cid, aid)
                weights = cln_model.params[wi].data.cpu().numpy()
                eta = cln_model.params[ei].item()

                # get the chosen x variables
                cmp_name = cln_model.atom_cmp[cid][aid]
                chosen = cln_model.atom_feat[wi]
                xvars = [x[chosen]]
                if None in xvars:
                    smt_clause_atom[cid].append(None)
                else:
                    lhs = sum_func([weights[i]*xvars[i] for i in range(len(xvars))])
                    smt_clause_atom[cid].append( \
                        compare(cmp_name, lhs, eta))

        if cid not in cln_model.del_cids:
            # get leaf for the clause
            li = 'l_%s' % cid
            leaf_val = cln_model.params[li].item()

            if leaf_val > 0:
                max_leaf_val = leaf_val
                min_leaf_val = 0
            else:
                max_leaf_val = 0
                min_leaf_val = leaf_val

            if False in smt_clause_atom[cid]:
                smt_and_clause[cid] = False
                leaf_ret[cid] = [0, 0, 0]
            elif None in smt_clause_atom[cid]:
                smt_and_clause[cid] = None
                leaf_ret[cid] = [None, min_leaf_val, max_leaf_val]
            else:
                smt_and_clause[cid] = True
                leaf_ret[cid] = [leaf_val, leaf_val, leaf_val]
    return smt_clause_atom, smt_and_clause, leaf_ret

