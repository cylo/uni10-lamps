import numpy as np
from pyUni10 import *
import shared

def load_networks(net_dir=shared.NETDIR):
    nets = {}
    nets["decision_fn_left"] = Network(net_dir + "/decision_fn_left.net")
    nets["decision_fn_right"] = Network(net_dir + "/decision_fn_right.net")
    nets["label_projection"] = Network(net_dir + "/label_projection.net")
    nets["measurement"] = Network(net_dir + "/measurement.net")
    return nets

def true_label_vec(true_label, dim_label):
    tl = UniTensor([Bond(BD_IN, dim_label)])
    elem = np.zeros((dim_label)); elem[true_label] = 1.0
    tl.setElem(elem)
    return tl

def l2_norm(vec):
    return contract(vec, vec, False)[0]
