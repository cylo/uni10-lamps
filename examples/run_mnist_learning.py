import time
import matplotlib.pyplot as plt
from keras.datasets import mnist

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from lamps.lamps import *

# ========================================================================

if __name__ == '__main__':

    from var_mnist_learning import *
    shared.OLD_VER = old_ver

    print "Training MNIST with",
    print "bond_dim_max={}, sweep={}, num_train={}, num_test={}, ...".format(m, sw, Ntrain, Ntest)

    (x0_train, y0_train), (x0_test, y0_test) = mnist.load_data()
    train_img = preprocess_grayscale(x0_train[:Ntrain], shrink_factor=sf)
    test_img = preprocess_grayscale(x0_test[:Ntest], shrink_factor=sf)
    train_set = ImageDataSet(train_img, y0_train[:Ntrain], bdry_dummy=True, func=grayscale_to_spinor)
    test_set = ImageDataSet(test_img, y0_test[:Ntest], bdry_dummy=True, func=grayscale_to_spinor)

    networks = load_networks()
    w = ClassifierMPS(train_set.wd, train_set.ht, networks, chi_max=m, dim_label=l, dim_virt=1)
    w.uniform(0.95/float(w.dv))

    start = time.time()
    optim = GDOptimizer(train_set, w, networks, mproc=procs)
    optim.gradient_descent(sweeps=2, normalize=True, site_term=1)  # precondition
    optim.gradient_descent(sweeps=sw-2, normalize=Ntrain, site_term=train_img[0].size//2)
    print "Training done.\nTime elapsed = {} seconds.\n".format(time.time()-start)

    print "Train accuracy = {}".format(w.accuracy(train_set, refresh_phi_rn=False))
    print "Test accuracy = {}".format(w.accuracy(test_set, refresh_phi_rn=True))

    if sav_path:
        w.save(sav_path)
    if show_feat_map:
        for i in xrange(l):
            ig = w.feature_map(i, func=spinor_to_grayscale)
            #ig = enhance_grayscale(ig, 0.5, 10.)
            plt.imshow(ig, cmap='Greys')
            plt.show()
