#!/usr/bin/env python
# coding: utf-8

import sys
import argparse
import json
import math
import numpy as np
import torch
import xgboost as xgb
from collections import defaultdict
from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from clnmodel import CLNModel
from util import *
from operator import itemgetter
from projector import *
from torch.utils.data import TensorDataset, DataLoader
from gurobipy import *
#from social_attack_ilp import MAX_INT32, ILPModel
from attack_ilp import MAX_INT32, ILPModel
from clnmodel import GUARD_VAL

def parse_args():
    parser = argparse.ArgumentParser(description='Train CLN model.')
    parser.add_argument('--train', '--train_data', type=str, help='train data file name.', required=True)
    parser.add_argument('--validation', '--validation_data', type=str, help='validation data file name.', required=True)
    parser.add_argument('--test', '--test_data', type=str, help='test data file name.', required=True)
    parser.add_argument('--nlabels', type=int, default=None, help='number of labels.', required=True)
    parser.add_argument('--intfeat', type=str, default=None, help='the list of feature indices that are integers.')
    parser.add_argument('-n', '--nfeat', type=int, help='number of features.', required=True)
    parser.add_argument('-z', '--zero_start', default=True, help='whether the feature starts from 0.', action='store_true')
    parser.add_argument('-e', '--epoch', type=int, default=1, help='number of training epoches')
    parser.add_argument('--num_clauses', type=int, default=0, help='number of clauses in the DNF', required=False)
    parser.add_argument('--min_atoms', type=int, default=1, help='minimum number of atoms in a conjunctive clause', required=False)
    parser.add_argument('--max_atoms', type=int, default=1, help='maximum number of atoms in a conjunctive clause', required=False)
    parser.add_argument('--structure', type=str, default=None, help='use a trained tree model structure to initialize the model.', required=False)
    parser.add_argument('--init', action="store_true", default=False, help='initialize the parameters of the structure from the existing model.', required=False)
    parser.add_argument('--init_b', type=int, default=500, help='initialize B used in CLN.', required=False)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.', required=False)
    parser.add_argument('--schedule', type=int, default=0, help='schedule B: increment per epoch', required=False)
    parser.add_argument('--same', action="store_true", default=False, help='use the same training parameters for the same atoms in the structure from the existing model.', required=False)
    parser.add_argument('--header', type=str, help='csv containing field names for features.', required=False)
    parser.add_argument('--load_model_path', type=str, default = None, help='load model path.')
    parser.add_argument('--save_model_path', type=str, help='model path to save the smooth CLN model', default="model.h5")
    parser.add_argument('--save_json', action="store_true", default=False)
    parser.add_argument('--just_save', action="store_true", default=False)
    parser.add_argument('--just_test', action="store_true", default=False)
    parser.add_argument('--cutoff', type=float, default=None, help='the prediction cutoff to evaluate the model', required=False)
    parser.add_argument('--fpr', type=float, default=None, help='the prediction cutoff at a positive rate to evaluate the model', required=False)
    parser.add_argument('--resume', action="store_true", default=False)
    parser.add_argument('--default_lo', type=float, default=0, help='default lower bound for features', required=False)
    parser.add_argument('--default_hi', type=float, default=None, help='default upper bound for features', required=False)
    parser.add_argument('-b', type=float, default=100, help='maximum lp norm bound.', required=False)
    parser.add_argument('--robust', action="store_true", default=False)
    parser.add_argument('--subprop', action="store_true", default=False)
    parser.add_argument('--monotonicity', type=str, default=None, help='list of feature indices for the monotone features.', required=False)
    parser.add_argument('--monotonicity_dir', type=str, default=None, help='the direction of monotone features.', required=False)
    parser.add_argument('--stability', type=str, default=None, help='list of feature indices for the stable features.', required=False)
    parser.add_argument('--stability_th', type=float, default=0.2, help='the constant threshold for stable features.', required=False)
    parser.add_argument('--eps', type=float, default=None, help='small modification distance property.', required=False)
    parser.add_argument('--C', type=float, default=None, help='lipschitz constant for small modification distance property.', required=False)
    parser.add_argument('--featmax', type=str, default = None, help='csv file for max feature values.', required=False)
    parser.add_argument('--lowcost', type=str, default=None, help='dict of feature bounds for the low cost feature property.', required=False)
    parser.add_argument('--lowcost_th', type=float, default=0.9, help='the confidence threshold of low cost features.', required=False)
    parser.add_argument('--redundancy', type=str, default=None, help='sets of features for redundancy. \
                        e.g., [{0:(6, None), 1:(None, None)}, {8:(None, None), 9:(None, None)}, \
                            {10:(None, None), 11:(None, None)}, {12:(None, None), 13:(None, None)}] \
                            says that [0, 1], [8, 9], [10, 11], [12, 13] are redundant of each other.')
    parser.add_argument('--size', type=int, default=1024, help='size of mini batch in training', required=False)
    # xgboost parameters
    parser.add_argument('--num_boost_round', type=int, help='Number of trees.', required=False)
    parser.add_argument('--max_depth', type=int, help='Maximum number of depth for each tree.', required=False)
    parser.add_argument('--scale_pos_weight', type=float, default=1, help='scale_pos_weight parameter.', required=False)
    parser.add_argument('--loss_weight', action="store_true", default=False)
    parser.add_argument('--add', type=str, choices=['path', 'tree'], help='increment the model by path or by tree.', required=False)
    parser.add_argument('--randfree', action="store_true", help='schedule the features to use for each boosting round for social honeypot dataset', default=False)
    parser.add_argument('--fixlast', action="store_true", default=False)
    parser.add_argument('--demo', action="store_true", default=False)
    parser.add_argument('--exfeat', type=str, default=None, help='exclude the list of features', required=False)
    return parser.parse_args()

def test_stats(model, X, Y, cutoff = None, fpr = None):

    x_test_tensor = torch.from_numpy(X).float()
    y_test_tensor = torch.from_numpy(Y).float()
    test_data = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_data, batch_size=args.size, shuffle=True)
    
    all_preds = []
    all_y = []
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.cuda()
        all_y.extend(y_batch.numpy())
        y_batch = y_batch.cuda()
        scores = model(x_batch, y_batch)
        all_preds.extend(scores.cpu().detach().numpy())

    print('CLN performance:')
    #print(type(all_y))
    #print(all_y)
    #print(type(all_preds))
    #print(all_preds)
    th, tpr, fpr, auc = print_stats(all_y, all_preds, cutoff = cutoff, fpr = fpr)
    return tpr, auc

def print_stats(all_y, all_preds, cutoff = None, fpr = None):
    if cutoff != None:
        th, tpr, fpr, auc = stats_at_cutoff(all_y, all_preds, cutoff = cutoff)
    elif fpr != None:
        th, tpr, fpr, auc = stats_at_fpr(all_y, all_preds, fpr = fpr)
    else:
        # default 0.5 cutoff
        th, tpr, fpr, auc = stats_at_cutoff(all_y, all_preds, cutoff = 0.5)
    return th, tpr, fpr, auc

def get_model_stats(threshold, y, preds):
    y_pred = [1 if p >= threshold else 0 for p in preds]
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    #print(tp, tn, fp, fn)
    acc = (tp+tn)/float(tp+tn+fp+fn)
    fpr = fp/float(fp+tn)
    tpr = tp/float(tp+fn)
    tnr = tn/float(fp+tn)
    fnr = fn/float(fn+tp)
    precision = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    return tpr, tnr, fpr, fnr, acc, precision

def stats_at_fpr(y_test, preds, fpr = None):
    # AUC
    fps, tps, thresholds = metrics.roc_curve(y_test, preds, drop_intermediate = False)
    auc = metrics.auc(fps, tps)
    #print("\nAUC: {:.5f}".format(auc))

    print("\nperf\tThreshold\tTPR\tTNR\tFPR\tFNR\tAcc\tPrec\tAUC\tF1")
    if fpr == None:
        tpr, tnr, fpr, fnr, acc, precision = get_model_stats(0.5, y_test, preds)
        print('perf\t0.5\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.5f\t%.3f\n' % \
              (tpr, tnr, fpr, fnr, acc, precision, auc, f1_score(precision, tpr)))
        prev_th = 0.5
    else:
        prev_tps, prev_fps, prev_th = None, None, None
        first = True
        # print Acc, TPR, FNR when FPR <= fpr
        for j in range(len(thresholds)):
            if fps[j] > fpr and first:
                first = False
                #print('%s\t%s\t%s' % (prev_th, prev_tps, prev_fps))
                tpr, tnr, fpr, fnr, acc, precision = get_model_stats(prev_th, y_test, preds)
                print('perf\t%.7f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.5f\t%.3f\n' % \
                      (prev_th, tpr, tnr, fpr, fnr, acc, precision, auc, f1_score(precision, tpr)))
                break
            prev_th, prev_tps, prev_fps = thresholds[j], tps[j], fps[j]
    return prev_th, tpr, fpr, auc

def stats_at_cutoff(y_test, preds, cutoff = None):
    # AUC
    fps, tps, thresholds = metrics.roc_curve(y_test, preds, drop_intermediate = False)
    auc = metrics.auc(fps, tps)

    print("\nperf\tThreshold\tTPR\tTNR\tFPR\tFNR\tAcc\tPrec\tAUC\tF1")
    tpr, tnr, fpr, fnr, acc, precision = get_model_stats(cutoff, y_test, preds)
    print('perf\t%.7f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.5f\t%.3f\n' % \
            (cutoff, tpr, tnr, fpr, fnr, acc, precision, auc, f1_score(precision, tpr)))
    return cutoff, tpr, fpr, auc

def discrete_test_stats(model, X, Y, label_cnt = False, save_scores = False, cutoff = None, fpr = None):
    x_test_tensor = torch.from_numpy(X).float()
    y_test_tensor = torch.from_numpy(Y).float()
    test_data = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_data, batch_size=args.size, shuffle=True)

    all_preds = []
    all_y = []
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.cuda()
        y_batch = y_batch.numpy()
        scores = model.discrete_pred(x_batch, y_batch, label_cnt, save_scores)
        all_preds.extend(scores)
        all_y.extend(y_batch)

    print('Discrete performance:')
    th, tpr, fpr, auc = print_stats(all_y, all_preds, cutoff = cutoff, fpr = fpr)
    return th, tpr, fpr, auc

def train(cln_model, x_train, y_train, epoch, projector, start_cid):
    opt = torch.optim.Adam(list(cln_model.parameters()), lr=args.lr)
    # LR DECAY
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(opt, lambda epoch: 0.95)
    loss_trace = []
    cnt = 0

    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_data, batch_size=args.size, shuffle=True)

    for x_batch, y_batch in train_loader:
        opt.zero_grad()
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        y_pred = cln_model(x_batch, y_batch, train_cnt = False).cuda()
        leaves = []
        for cid in cln_model.all_cids:
            li = 'l_%s' % cid
            leaves.append(cln_model.params[li])

        #loss = criterion(y_pred, y_batch) + torch.norm(torch.tensor(leaves), p=1)
        if args.loss_weight == True and args.scale_pos_weight != 1.0:
            weight = torch.from_numpy(np.array([args.scale_pos_weight if i == 1 else 1.0 for i in y_batch])).cuda()
        else:
            weight = torch.ones(x_batch.shape[0]).cuda()
        loss = torch.nn.functional.binary_cross_entropy(y_pred, y_batch, weight=weight)
        loss_trace.append(loss.item())
        loss.backward(retain_graph=True)

        opt.step()

        # enforce constraints
        # assume that leaf values are not constants
        if args.robust == True:
            #if epoch != 0:
            in_leaves = {}
            for cid in cln_model.all_cids:
                li = 'l_%s' % cid
                leaf_val = cln_model.params[li].item()
                in_leaves[cid] = leaf_val
            out_leaves = projector.project(in_leaves)
            # update leaf variables
            for cid, leaf_val in out_leaves.items():
                if cid < start_cid:
                    continue
                li = 'l_%s' % cid
                cln_model.params[li] = torch.nn.Parameter(torch.tensor(leaf_val).cuda())

        cnt += 1
    return loss_trace

'''
both regular training and property training
'''
def train_property(args, cln_model, start_cid, x_train, y_train, \
    x_test, y_test, x_validation, y_validation, iteration, path, total_rounds, prev_saves, featmax):
    
    if args.robust == True:
        oracle = ILPModel(cln_model, start_cid, cln_model.last_cid, int_indices, args.nfeat, args.default_lo)
        projector = Constraint_Projector(cln_model)
    else:
        oracle = None
        projector = None
    
    print("test performance before training")
    test_stats(cln_model, x_test, y_test, cutoff = args.cutoff, fpr = args.fpr)
    sys.stdout.flush()
    print("discrete test performance before training")
    th, tpr, fpr, auc = discrete_test_stats(cln_model, x_test, y_test, False, cutoff = args.cutoff, fpr = args.fpr)
    sys.stdout.flush()

    # 20 epochs to warm up
    if args.init == False:
        for epoch in range(20):
            print('B:', cln_model.B)
            sys.stdout.flush()
            loss_trace = train(cln_model, x_train, y_train, epoch, projector, start_cid)
            loss_trace = np.array(loss_trace)
            print(loss_trace)
            sys.stdout.flush()
            print('loss_trace: mean {:.4f}, min {:.4f}, max {:.4f}'.format(loss_trace.mean(),\
                            loss_trace.min(), loss_trace.max()))
        
        cln_model.reset_label_cnt()
        print('\n======= after warm up =======\n')
        print("validation performance")
        test_stats(cln_model, x_validation, y_validation, cutoff = args.cutoff, fpr = args.fpr)
        print("discrete validation performance")
        th, tpr, fpr, auc = discrete_test_stats(cln_model, x_validation, y_validation, label_cnt = True, save_scores = True, cutoff = args.cutoff, fpr = args.fpr)
        cln_model.save_numpy_scores()
        cln_model.compute_acc_gain()
        cln_model.compute_entropy()
        cln_model.compute_information_gain(y_validation)

        cln_model.prettyprint()

    # exponential backoff timeout value for all epochs
    time_val = 30
    
    # CEGIS loop to add constraint, and train CLN
    for epoch in range(args.epoch):
        # change the counter when I have more properties
        cur_counter = int(args.monotonicity != None) + int(args.stability != None) \
            + int(args.lowcost != None) \
            + int(args.eps != None) + int(args.redundancy != None)
        global_counter = int(args.monotonicity != None) + int(args.stability != None) \
            + int(args.lowcost != None) \
            + int(args.eps != None) + int(args.redundancy != None)
        # print("*** DEBUG: sat_monotonicity:", sat_monotonicity)
        # print("*** DEBUG: sat_stability:", sat_stability)
        # print("*** DEBUG: sat_lowcost:", sat_lowcost)

        has_constraint = False

        # global properties
        if args.robust == True and global_counter != 0:
            for k, prop_param in enumerate([args.monotonicity, args.stability, \
                    args.lowcost, args.eps, args.redundancy]):
                if prop_param == None:
                    continue
                if k == 0:
                    prop_name = 'monotonicity'
                    prop_index_list = eval(args.monotonicity)
                    monotone_direction = eval(args.monotonicity_dir)
                elif k == 1:
                    prop_name = 'stability'
                    prop_index_list = eval(args.stability)
                    if args.subprop:
                        stable_threshold = args.stability_th / float(total_rounds)
                    else:
                        stable_threshold = args.stability_th
                elif k == 2:
                    prop_name = 'lowcost'
                    lowcost_dict = eval(args.lowcost)
                    prop_index_list = list(lowcost_dict.keys())
                    confidence_threshold = args.lowcost_th
                elif k == 3:
                    prop_name = 'eps'
                    constant = args.C
                    eps = args.eps
                    maxdiff = eps * constant
                    if args.subprop:
                        maxdiff = maxdiff / float(total_rounds)
                        eps = eps / float(total_rounds)
                    # just a placeholder
                    prop_index_list = [-1]
                elif k == 4:
                    prop_name = 'redundancy'
                    lowcost_array = eval(args.redundancy)
                    confidence_threshold = args.lowcost_th
                    cutoff = 0.5
                    prop_index_list = lowcost_array
                    prop_lower = []
                    prop_upper = []
                    for lowcost_dict in lowcost_array:
                        fi_list = []
                        lower_list = []
                        upper_list = []
                        for lowcost_index, bounds in lowcost_dict.items():
                            lower, upper = bounds
                            fi_list.append(lowcost_index)
                            lower_list.append(lower)
                            upper_list.append(upper)
                        prop_lower.append(lower_list)
                        prop_upper.append(upper_list)
                        #attack_model.global_attack(st, ed, 'redundancy', fi_list, lower_list, upper_list, cutoff, confidence_threshold)

                # save index to the prop_index_list for timed out attacks
                time_out_list = []
                # for every fi, assemble attack_args according to the prop_name
                prop_counter = len(prop_index_list)
                print('\n======= attacking %s =======\n' % prop_name)
                for i, fi in enumerate(prop_index_list):
                    if k == 0:
                        attack_args = [[fi], monotone_direction[i]]
                        clause_args = []
                    elif k == 1:
                        attack_args = [[fi], stable_threshold]
                        clause_args = [stable_threshold]
                    elif k == 2:
                        lower, upper = lowcost_dict[fi]
                        attack_args = [[fi], lower, upper, th, confidence_threshold]
                        cname = 'c_e%d_fid%d' % (epoch, fi)
                        clause_args = [th, confidence_threshold, cname]
                    elif k == 3:
                        attack_args = [list(range(args.nfeat)), eps, maxdiff, featmax]
                        clause_args = [maxdiff]
                    elif k == 4:
                        attack_args = [lowcost_array[i].keys(), prop_lower[i], prop_upper[i], cutoff, confidence_threshold]
                        cname = 'c_e%d_fid%s' % (epoch, list(lowcost_array[i].keys()))
                        clause_args = [th, confidence_threshold, cname]
                    else:
                        attack_args = []
                        clause_args = []
                    changed_atoms, pred_scores, ret = oracle.global_attack(start_cid, cln_model.last_cid, prop_name, *attack_args)
                    # the things after the attack are repeated.
                    if ret != GRB.INFEASIBLE and ret != GRB.TIME_LIMIT and len(changed_atoms) == 0 and has_constraint == False:
                        print('oracle.grb set IntFeasTol to 1e-9')
                        oracle.grb.setParam('IntFeasTol', 1e-9)
                        changed_atoms, pred_scores, ret = oracle.global_attack(start_cid, cln_model.last_cid, prop_name, *attack_args)
                    if ret == GRB.INFEASIBLE:
                        prop_counter -= 1
                        print('\n======= feature %s: no more violations against %s =======\n' % (fi, prop_name))
                        if prop_counter == 0:
                            global_counter -= 1
                            cur_counter -= 1
                    elif ret == GRB.TIME_LIMIT:
                        print('attack exceeded time limit')
                        time_out_list.append(i)
                    else:
                        addnum = cln_model.add_constraint(projector, changed_atoms, pred_scores, prop_name, \
                            *clause_args)
                        if addnum == 1:
                            has_constraint = True
                            print('\n======= feature %s: after adding a constraint for %s =======\n' % (fi, prop_name))
                            cln_model = cln_model.cuda()
                
                # go over timed out ones until we find one constraint
                # exponential back off
                # didn't add constraint and not all attacks are infeasible

                last_add = 0
                while has_constraint == False and prop_counter != 0 and len(time_out_list) != 0:
                    print('\n======= attacking timed out ones %s =======\n' % prop_name)
                    time_val *= 2
                    oracle.grb.setParam('TimeLimit', time_val)
                    print('oracle.grb set TimeLimit to', time_val)
                    i = time_out_list.pop()
                    fi = prop_index_list[i]
                    if k == 0:
                        attack_args = [[fi], monotone_direction[i]]
                        clause_args = []
                    elif k == 1:
                        attack_args = [[fi], stable_threshold]
                        clause_args = [stable_threshold]
                    elif k == 2:
                        lower, upper = lowcost_dict[fi]
                        attack_args = [[fi], lower, upper, th, confidence_threshold]
                        cname = 'c_e%d_fid%d' % (epoch, fi)
                        clause_args = [th, confidence_threshold, cname]
                    elif k == 3:
                        attack_args = [list(range(args.nfeat)), eps, maxdiff, featmax]
                        clause_args = [maxdiff]
                    elif k == 4:
                        attack_args = [lowcost_array[i].keys(), prop_lower[i], prop_upper[i], cutoff, confidence_threshold]
                        cname = 'c_e%d_fid%s' % (epoch, list(lowcost_array[i].keys()))
                        clause_args = [th, confidence_threshold, cname]
                    else:
                        attack_args = []
                        clause_args = []
                    changed_atoms, pred_scores, ret = oracle.global_attack(start_cid, cln_model.last_cid, prop_name, *attack_args)
                    if ret != GRB.INFEASIBLE and ret != GRB.TIME_LIMIT and len(changed_atoms) == 0 and has_constraint == False:
                        print('oracle.grb set IntFeasTol to 1e-9')
                        oracle.grb.setParam('IntFeasTol', 1e-9)
                        changed_atoms, pred_scores, ret = oracle.global_attack(start_cid, cln_model.last_cid, prop_name, *attack_args)
                    if ret == GRB.INFEASIBLE:
                        prop_counter -= 1
                        print('\n======= feature %s: no more violations against %s =======\n' % (fi, prop_name))
                        if prop_counter == 0:
                            global_counter -= 1
                            cur_counter -= 1
                    elif ret == GRB.TIME_LIMIT:
                        print('attack exceeded time limit')
                        time_out_list.append(i)
                    else:
                        last_add = cln_model.add_constraint(projector, changed_atoms, pred_scores, prop_name, \
                            *clause_args)
                        if last_add == 1:
                            has_constraint = True
                            print('\n======= feature %s: adding a constraint for %s =======\n' % (fi, prop_name))
                            cln_model = cln_model.cuda()
                # reset to 30 seconds for the next property / round
                #if last_add == 1:
                #    print('oracle.grb set TimeLimit to 30')
                #    oracle.grb.setParam('TimeLimit', 30)

            #oracle.update(cln_model)
            #projector.update(cln_model)

        # not robust training or not all properties are sat
        if args.robust == False or cur_counter > 0 or has_constraint == True:
            #do CLN training
            # train one epoch over the current structure of cln_model
            # schedule B
            cln_model.B += args.schedule
            print('B:', cln_model.B)
            sys.stdout.flush()
            loss_trace = train(cln_model, x_train, y_train, epoch, projector, start_cid)
            loss_trace = np.array(loss_trace)
            print(loss_trace)
            sys.stdout.flush()
            print('loss_trace: mean {:.4f}, min {:.4f}, max {:.4f}'.format(loss_trace.mean(),\
                            loss_trace.min(), loss_trace.max()))
        else:
            # TODO: add some regular training at the end
            if args.robust == True and cur_counter == 0:
                pass


        # check discrete model stat
        # save the current best model
        cln_model.reset_label_cnt()
        print('\n======= after epoch %d =======\n' % epoch)
        print("validation performance")
        test_stats(cln_model, x_validation, y_validation, cutoff = args.cutoff, fpr = args.fpr)
        print("discrete validation performance")
        th, tpr, fpr, auc = discrete_test_stats(cln_model, x_validation, y_validation, label_cnt = True, save_scores = True, cutoff = args.cutoff, fpr = args.fpr)
        cln_model.save_numpy_scores()
        cln_model.compute_acc_gain()
        cln_model.compute_entropy()
        cln_model.compute_information_gain(y_validation)

        # debug the state of the model
        # print the model again for demo purpose
        if args.demo:
            cln_model.prettyprint()

        sys.stdout.flush()

        # save the model for regular training or all properties are sat
        if cur_counter == 0:
            cur_model_path = '%s_b%d_p%d_e%d.pth' % (args.save_model_path.split('.pth')[0], iteration, path, epoch)
            # save the model with best validation acc
            if auc > cln_model.best_val_auc:
                cln_model.best_model_path = cur_model_path
                cln_model.best_val_auc = auc
                cln_model.best_epoch = epoch
            torch.save(cln_model, cur_model_path)
            prev_saves.append(cur_model_path)
            #cln_model.prettyprint()
            print("CLN model AUC", auc, "saved to", cur_model_path, "and", cln_model.save_json_path, "\n")
            cln_model.save_as_json()
            
            # if it's regular training, continue to the next epoch
            if args.robust == True:
                break
        
    sys.stdout.flush()
    return th

def main(args):
    if args.train.endswith(".libsvm"):
        # read train test data
        x_train, y_train = load_svmlight_file(args.train,
                                           n_features=args.nfeat,
                                           multilabel=(args.nlabels != 2),
                                           zero_based=args.zero_start)
        x_train = x_train.toarray().astype(np.float32)
        y_train = y_train.astype(np.int)
    elif args.train.endswith(".csv"):
        x_train = np.loadtxt(args.train, delimiter=',', usecols=list(range(1, args.nfeat+1))).astype(np.float32)
        y_train = np.loadtxt(args.train, delimiter=',', usecols=0).astype(np.int)
    else:
        print("file format not supported yet.")
        exit()

    if args.validation.endswith(".libsvm"):
        x_validation, y_validation = load_svmlight_file(args.validation,
                                           n_features=args.nfeat,
                                           multilabel=(args.nlabels != 2),
                                           zero_based=args.zero_start)
        x_validation = x_validation.toarray().astype(np.float32)
        y_validation = y_validation.astype(np.int)
    elif args.validation.endswith(".csv"):
        x_validation = np.loadtxt(args.validation, delimiter=',', usecols=list(range(1, args.nfeat+1))).astype(np.float32)
        y_validation = np.loadtxt(args.validation, delimiter=',', usecols=0).astype(np.int)
    else:
        print("file format not supported yet.")
        exit()

    if args.test.endswith(".libsvm"):
        x_test, y_test = load_svmlight_file(args.test,
                                           n_features=args.nfeat,
                                           multilabel=(args.nlabels != 2),
                                           zero_based=args.zero_start)
        x_test = x_test.toarray().astype(np.float32)
        y_test = y_test.astype(np.int)
    elif args.test.endswith(".csv"):
        x_test = np.loadtxt(args.test, delimiter=',', usecols=list(range(1, args.nfeat+1))).astype(np.float32)
        y_test = np.loadtxt(args.test, delimiter=',', usecols=0).astype(np.int)
    else:
        print("file format not supported yet.")
        exit()

    # read the list of feature indices that are Int
    global int_indices
    int_indices = set([])
    if args.intfeat != None:
        infile = json.load(open(args.intfeat, 'r'))
        int_indices = infile['indices']

    save_json_path = '%s.json' % args.save_model_path.split('.pth')[0]

    # initialize B for CLN
    init_b = args.init_b
    print('B:', args.init_b)

    # load a fixed structure if specified
    if args.structure != None:
        print("\n======= Initialize CLN model =======\n")
        cln_model = CLNModel(args.header, args.num_clauses, args.min_atoms, args.max_atoms, \
                            args.structure, save_json_path, -1, args.init, args.same, int_indices, args.nfeat, args.nlabels, \
                            x_train, B=init_b, negate=False)
        cln_model.prettyprint()
        cln_model = cln_model.cuda()
        cln_model.reset_label_cnt()
    if args.load_model_path != None:
        print("\n======= Load CLN model =======\n")
        cln_model = torch.load(args.load_model_path)
        cln_model = cln_model.cuda()
        cln_model.reset_label_cnt()
    
    if args.just_test == True:
        print("test performance before training")
        test_stats(cln_model, x_test, y_test, cutoff = args.cutoff, fpr = args.fpr)
        sys.stdout.flush()
        print("discrete test performance before training")
        th, tpr, fpr, auc = discrete_test_stats(cln_model, x_test, y_test, False, cutoff = args.cutoff, fpr = args.fpr)
        sys.stdout.flush()
        return

    if args.featmax != None:
        featmax = np.loadtxt(args.featmax, delimiter=',', usecols=list(range(args.nfeat))).astype(np.float32)
    else:
        featmax = None

    prev_saves = []
    if args.robust == False:
        train_property(args, cln_model, 0, x_train, y_train, \
            x_test, y_test, x_validation, y_validation, 1, 0, 1, prev_saves, featmax)
        if cln_model.best_model_path != None:
            # print best model performance and delete other previous saves
            best_model = torch.load(cln_model.best_model_path)
            print('\n======= after training =======\n')
            # test
            print('best model:', cln_model.best_model_path)
            print("test performance for the best model")
            test_stats(best_model, x_test, y_test, cutoff = args.cutoff, fpr = args.fpr)
            print("discrete test performance for the best model")
            discrete_test_stats(best_model, x_test, y_test, cutoff = args.cutoff, fpr = args.fpr)
            # if we can finish all boosting rounds
            best_model.save_json_path = cln_model.best_model_path.split('.pth')[0] + '.json'
            best_model.save_as_json()
            print("model json with best validation AUC save to", best_model.save_json_path)
        return
    
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    param = {'max_depth': args.max_depth, 'objective': 'binary:logistic', 'eta': 1, \
            'eval_metric': 'auc', 'scale_pos_weight': args.scale_pos_weight}
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    if args.exfeat != None:
        fw = np.ones(shape=(args.nfeat,))
        exfeat_list = eval(args.exfeat)
        for i in exfeat_list:
           fw[i] = 0
        dtrain.set_info(feature_weights=fw)
        print('fw:', fw)
        # feature sampling
        param['colsample_bytree'] = 1 - len(exfeat_list)/args.nfeat

    if args.structure == None and args.load_model_path == None:
        # iteration 0
        xgbmodel = xgb.train(param, dtrain, num_boost_round = 1, \
                evals = evallist)
        preds = xgbmodel.predict(dtest)
        print("\n======= Boosting round 0 =======\n")
        print("test performance after boosting")
        print_stats(y_test, preds, cutoff = args.cutoff, fpr = args.fpr)
        # save model json
        print("save model json to:", save_json_path)
        xgbmodel.save_model(save_json_path)
        
        print("\n======= Initialize CLN model =======\n")
        cln_model = CLNModel(args.header, args.num_clauses, args.min_atoms, args.max_atoms, \
                            save_json_path, save_json_path, 0, args.init, args.same, int_indices, args.nfeat, args.nlabels, \
                            x_train, B=init_b, negate=False)
    elif args.load_model_path != None:
        print("\n======= Load CLN model =======\n")
        cln_model = torch.load(args.load_model_path)
        print("cln model loaded from", args.load_model_path)
        cln_model.save_json_path = save_json_path

        # TODO: remove DEBUG
        cln_model.prettyprint()
        cln_model.save_as_json()
        xgbmodel = xgb.train(param, dtrain, num_boost_round = 1, \
                evals = evallist, xgb_model = save_json_path)
        print("save model json to:", save_json_path)
        xgbmodel.save_model(save_json_path)

        cln_model.json_content = json.load(open(cln_model.save_json_path, 'r'))
        parsed_content = parse_json(cln_model.json_content)
        last_tree_paths = parsed_content[-1]
        # add by tree
        if args.add == 'tree':
            # for the newly boosted tree, add the new tree
            print("\n======= Add a new tree to the CLN model =======\n")
            for pathid, path in enumerate(last_tree_paths):
                all_atoms, leaf_value = path
                cln_model.add_one_path(all_atoms, leaf_value)
            cln_model = cln_model.cuda()

            cln_model.reset_label_cnt()
            cln_model.prettyprint()

        cln_model.save_as_json()
        print('train properties')
        start_cid = 0
        last_cutoff = train_property(args, cln_model, start_cid, x_train, y_train, \
            x_test, y_test, x_validation, y_validation, 1, 0, 1, prev_saves, featmax)
        return

    cln_model = cln_model.cuda()
    cln_model.reset_label_cnt()
    cln_model.prettyprint()
    sys.stdout.flush()

    if args.just_save == True:
        # save cln_model
        cln_model.save_as_json()
        torch.save(cln_model, args.save_model_path)
        print("initialized model save to", args.save_model_path)
        sys.stdout.flush()
        return
    
    start_cid = 0
    # start_cid and total_rounds are set according to args.subprop
    if args.subprop:
        total_rounds = args.num_boost_round
    else:
        total_rounds = 1

    if args.randfree == True:
        save_args_eps = args.eps
        args.eps = None
        print('save_args_eps:', save_args_eps, 'args.eps:', args.eps)
    
    # TEST: don't attack the first one
    #if args.structure == None or args.init == False:
    #if True:
    # the first model will be attacked even if it is loaded
    if not args.fixlast or total_rounds == 1:
        last_cutoff = train_property(args, cln_model, start_cid, x_train, y_train, x_test, y_test, \
            x_validation, y_validation, 0, 0, total_rounds, prev_saves, featmax)
        
    # fix the current trees
    if args.subprop:
        cln_model.save_current_clauses()
    # need to save the json model every time for xgboost to read again for the next round    
    cln_model.save_as_json()

    # start_cid and total_rounds are set according to args.subprop
    if args.subprop:
        start_cid = cln_model.last_cid + 1

    # mono_list
    ex_mono = [0, 2, 3, 4, 10, 11]
    # lowcost_list
    ex_lowcost = [0, 1, 8, 9, 10, 11, 12, 13]
    ex_none = []
    all_ex = []
    if args.randfree == True:
        all_ex = [ex_lowcost, ex_lowcost, ex_mono, ex_mono]
    
    for iteration in range(1, args.num_boost_round):
        # schedule features for each boosting round
        if args.randfree == True:
            # currently this only works with the social honeypot 10 round specification
            if iteration <= args.num_boost_round - 2 and len(all_ex) > 0:
                exfeat_list = all_ex.pop(0)
                fw = np.ones(shape=(args.nfeat,))
                for i in exfeat_list:
                   fw[i] = 0
                dtrain.set_info(feature_weights=fw)
                print('fw:', fw)
                # feature sampling
                param['colsample_bytree'] = 1 - len(exfeat_list)/args.nfeat
            else:
                dtrain.set_info(feature_weights=None)
                param['colsample_bytree'] = 1
            if iteration == args.num_boost_round - 1:
                print('*** last round, try to restore args.eps ***')
                args.eps = save_args_eps
                print('args.eps: ', args.eps)

        # set dtrain and param accordingly
        xgbmodel = xgb.train(param, dtrain, num_boost_round = 1, \
                evals = evallist, xgb_model = save_json_path)
        
        feature_map = xgbmodel.get_fscore()
        print('*** feature_map ***')
        print(feature_map)

        preds = xgbmodel.predict(dtest)
        print("\n======= Boosting round %d =======\n" % iteration)
        print("test performance after boosting")
        print_stats(y_test, preds, cutoff = args.cutoff, fpr = args.fpr)
        # save model json
        print("save model json to:", save_json_path)
        xgbmodel.save_model(save_json_path)
        
        cln_model.json_content = json.load(open(cln_model.save_json_path, 'r'))
        parsed_content = parse_json(cln_model.json_content)
        last_tree_paths = parsed_content[-1]
        # add by tree
        if args.add == 'tree':
            # for the newly boosted tree, add the new tree
            print("\n======= Add a new tree to the CLN model =======\n")
            for pathid, path in enumerate(last_tree_paths):
                all_atoms, leaf_value = path
                cln_model.add_one_path(all_atoms, leaf_value)
            cln_model = cln_model.cuda()

            cln_model.reset_label_cnt()
            cln_model.prettyprint()
            sys.stdout.flush()

            if not args.fixlast or iteration == args.num_boost_round-1:
                print('train properties')
                last_cutoff = train_property(args, cln_model, start_cid, x_train, y_train, \
                    x_test, y_test, x_validation, y_validation, iteration, 0, total_rounds, prev_saves, featmax)
                # fix the current trees
                if args.subprop:
                    cln_model.save_current_clauses()
            
        # need to save the json model every time for xgboost to read again for the next round    
        cln_model.save_as_json()
        # start_cid and total_rounds are set according to args.subprop
        if args.subprop:
            start_cid = cln_model.last_cid + 1
    
    if cln_model.best_model_path != None:
        # print best model performance and delete other previous saves
        best_model = torch.load(cln_model.best_model_path)
        print('\n======= after training =======\n')
        # test
        print('best model:', cln_model.best_model_path)
        print("test performance for the best model")
        test_stats(best_model, x_test, y_test, cutoff = args.cutoff, fpr = args.fpr)
        #test_stats(best_model, x_test, y_test, cutoff = last_cutoff, fpr = None)
        print("discrete test performance for the best model")
        discrete_test_stats(best_model, x_test, y_test, cutoff = args.cutoff, fpr = args.fpr)
        #discrete_test_stats(best_model, x_test, y_test, cutoff = last_cutoff, fpr = None)
        best_model.save_json_path = cln_model.best_model_path.split('.pth')[0] + '.json'
        best_model.save_as_json()
        print("model json with best validation AUC save to", best_model.save_json_path)

    return

if __name__=='__main__':
    global args
    args = parse_args()
    main(args)

