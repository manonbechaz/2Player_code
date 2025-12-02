import torch
import numpy as np

def overall_accuracy(TP, TN, FP, FN):
    return (TP+TN)/(TP+TN+FP+FN)

def recall(TP, TN, FP, FN):
    return (TP+1e-10)/(TP+FN+1e-10)

def precision(TP, TN, FP, FN):
    return (TP+1e-10)/(TP+FP+1e-10)

def IoU(TP, TN, FP, FN):
    return (TP+1e-10)/(TP+FN+FP+1e-10)

def F1score(TP, TN, FP, FN):
    r = recall(TP, TN, FP, FN)
    p = precision(TP, TN, FP, FN)
    return 2*r*p/(r+p+1e-10)


def get_F1_from_pred(pred, lab):
        pr = (pred > 0.5).cpu().numpy()
        gt = (lab.data.int() > 0.5).cpu().numpy()
        
        tp = np.logical_and(pr, gt).sum()
        tn = np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
        fp = np.logical_and(pr, np.logical_not(gt)).sum()
        fn = np.logical_and(np.logical_not(pr), gt).sum()

        return F1score(tp,tn,fp,fn)