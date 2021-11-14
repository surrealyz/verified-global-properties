#!/usr/bin/env python
# coding: utf-8

import sys
import argparse
import numpy as np
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(description='Train a vanilla XGBoost GBDT model.')
    parser.add_argument('--train', '--train_data', type=str, help='train data file name.', required=True)
    parser.add_argument('--test', '--test_data', type=str, help='test data file name.', required=True)
    parser.add_argument('--nfeat', type=int, default=None, help='number of features.', required=True)
    parser.add_argument('--num_trees', type=int, help='Number of trees.', required=True)
    parser.add_argument('--max_depth', type=int, help='Maximum number of depth for each tree.', required=True)
    parser.add_argument('--model_name', type=str, help='Save the trained model.', required=True)
    parser.add_argument('--monotone', type=str, default=None, help='monotonic constraints string. 1/0/-1.', required=False)
    parser.add_argument('--exfeat', type=str, default=None, help='exclude the list of features', required=False)
    parser.add_argument('--scale_pos_weight', type=float, default=1, help='scale_pos_weight parameter.', required=False)
    parser.add_argument('--xgb_model', type=str, default=None, help='file name of stored xgb model to continue training.', required=False)
    return parser.parse_args()

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

def f1_score(precision, recall):
    return 2*precision*recall/float(precision+recall)

def main(args):
    #top10_idx = [2, 3, 5, 6, 7, 9, 10, 11, 12, 13]
    #top_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # read train test datasets
    X_train = np.loadtxt(args.train, delimiter=',', usecols=list(range(1, args.nfeat+1)))
    y_train = np.loadtxt(args.train, delimiter=',', usecols=0).astype(np.int)
    #X_train = X_train[:, top_idx]
    print(X_train.shape)
    X_test = np.loadtxt(args.test, delimiter=',', usecols=list(range(1, args.nfeat+1)))
    y_test = np.loadtxt(args.test, delimiter=',', usecols=0).astype(np.int)
    #X_test = X_test[:, top_idx]
    print(X_test.shape)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {'max_depth':args.max_depth, 'objective':'binary:logistic', 'eta':1, 'eval_metric':'auc'}
    # test other parameters
    # param['lambda'] = 3
    # param['eta'] = 0.3
    # param['gamma'] = 1
    #param['min_child_weight'] = 3
    #param['gamma'] = 1

    # remove exfeat using feature_weights
    # e.g., [0, 1, 3, 8, 9, 10, 11, 12, 13]
    if args.exfeat != None:
        fw = np.ones(shape=(args.nfeat,))
        exfeat_list = eval(args.exfeat)
        for i in exfeat_list:
           fw[i] = 0
        dtrain.set_info(feature_weights=fw)
        print('fw:', fw)
        # feature sampling
        param['colsample_bytree'] = 1 - len(exfeat_list)/args.nfeat

    if args.monotone != None:
        # param['monotone_constraints'] = "(0,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0)"
        # param['monotone_constraints'] = "(0,0,-1,0,-1,0,0,0,1,1,0,0,0,0,0)"
        param['monotone_constraints'] = str(args.monotone)

    param['scale_pos_weight'] = args.scale_pos_weight
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    xgbmodel = xgb.train(param, dtrain, num_boost_round = args.num_trees, \
                        evals = evallist, xgb_model = args.xgb_model)
    
    feature_map = xgbmodel.get_fscore()
    print(feature_map)

    if args.exfeat != None:
        for i in eval(args.exfeat):
            assert feature_map.get('f%d' % i, None) is None

    preds = xgbmodel.predict(dtrain)

    # AUC
    fps, tps, thresholds = metrics.roc_curve(y_train, preds, drop_intermediate = False)
    auc = metrics.auc(fps, tps)

    print("\nThreshold\tTPR\tTNR\tFPR\tFNR\tAcc\tPrec\tAUC\tF1\t\tscale_pos_weight")
    prev_tps, prev_fps, prev_th = None, None, None
    first_one = True
    first_onefive = True
    first_two = True
    # print Acc, TPR, FNR when FPR <= 1%, 1.5%, and 2%
    for j in range(len(thresholds)):
        if fps[j] > 0.015 and first_onefive:
            first_onefive = False
            tpr, tnr, fpr, fnr, acc, precision = get_model_stats(prev_th, y_test, xgbmodel.predict(dtest))
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.5f\t%.3f\t\t%.2f' % \
                  (prev_th, tpr, tnr, fpr, fnr, acc, precision, auc, f1_score(precision, tpr), args.scale_pos_weight))

        if fps[j] > 0.02 and first_two:
            first_two = False
            #print('%s\t%s\t%s' % (prev_th, prev_tps, prev_fps))
            tpr, tnr, fpr, fnr, acc, precision = get_model_stats(prev_th, y_test, xgbmodel.predict(dtest))
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.5f\t%.3f\t\t%.2f' % \
                  (prev_th, tpr, tnr, fpr, fnr, acc, precision, auc, f1_score(precision, tpr), args.scale_pos_weight))
        prev_th, prev_tps, prev_fps = thresholds[j], tps[j], fps[j]


    tpr, tnr, fpr, fnr, acc, precision = get_model_stats(0.5, y_test, xgbmodel.predict(dtest))
    print('%s\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.5f\t%.3f\t\t%.2f' % \
          (0.5, tpr, tnr, fpr, fnr, acc, precision, auc, f1_score(precision, tpr), args.scale_pos_weight))

    # y_pred = [1 if p > 0.5 else 0 for p in preds]
    # tpr, tnr, fpr, fnr, acc, precision = get_model_stats(0.5, y_test, preds)
    # print('TPR: ', tpr)
    # print('TNR: ', tnr)
    # print('Acc: ', acc)
    # print('FPR: ', fpr)
    xgbmodel.save_model("models/xgboost_models/%s.bin" % args.model_name)
    xgbmodel.save_model("models/xgboost_models/%s.json" % args.model_name)
    xgbmodel.dump_model("models/xgboost_models/%s.dumped.trees" % args.model_name)
    #xgbmodel.dump_model("/home/yz/code/Trees2SMTs/twitter_spam/xgboost_models/%s.json" % args.model_name, dump_format='json')
    return


if __name__=='__main__':
    global args
    args = parse_args()
    main(args)
