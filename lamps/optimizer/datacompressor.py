import os
import multiprocessing as mp
from itertools import izip

from mps.classifiermps import *
from utils import l2_norm
import shared

class DataSetCompressor(object):
    """
    """
    def __init__(self, image_set, clsfy_mps, networks, mproc=None):
        """"""
        self.__d = image_set
        self.__w = clsfy_mps
        self.net = networks
        self.__d.refresh_phi_rn(self.__w)
        if mproc:
            shared.USE_MP = True
            shared.PROCS = int(mproc)

    def site_tensor_rn(self, site):
        """"""
        wi = UniTensor()
        idx = site

        for img_data in self.__d:
            if site == self.__w.sl:
                net = self.net["sl_project"]
                net.putTensor("LAB", img_data.tl)
                net.putTensor("PHIL", img_data.phi_rn[idx-1])
                net.putTensor("PHII", img_data.phi[idx])
                net.putTensor("PHIR", img_data.phi_rn[idx+1])
                wsi = net.launch()
            else:
                net = self.net["site_project"]
                net.putTensor("PHIL", img_data.phi_rn[idx-1])
                net.putTensor("PHII", img_data.phi[idx])
                net.putTensor("PHIR", img_data.phi_rn[idx+1])
                wsi = net.launch()
            try:
                wi += wsi
            except:
                wi = wsi * 1.

        labw = [img_data.phi_rn[idx-1].label()[0], img_data.phi_rn[idx+1].label()[0], img_data.phi[idx].label()[0]]    
        if site == self.__w.sl: labw.insert(1, -10)
        wi.setLabel(labw)
        return wi

    def update_w(self, site_tensor, site, left_to_right=True, cutoff=1e-10, normalize=True, **kwargs):
        """"""
        inc = int(left_to_right)*2 - 1
        site_svd(site_tensor, self.__w[site+inc], self.__w.chi_max,
                 merge_sv_right=left_to_right, cutoff=cutoff, normalize=normalize, **kwargs)
        self.__w[site] = site_tensor * 1.

    def update_sl_rn(self, site, left_to_right=True):
        """"""
        idx = site
        if left_to_right:
            for s in xrange(self.__d.size):
                if idx == 0:
                    self.__d.phi_rn[s][idx] = self.__w[idx] * 1.;
                else:
                    self.__d.phi_rn[s][idx] = contract(self.__d.phi_rn[s][idx-1], self.__w[idx], False)
                self.__d.phi_rn[s][idx] = contract(self.__d.phi_rn[s][idx], self.__d.phi[s][idx], False)
                self.__d.tl[s].setLabel([-10])
                self.__d.phi_rn[s][idx] = contract(self.__d.phi_rn[s][idx], self.__d.tl[s], False)
        else:
            for s in xrange(self.__d.size):
                if idx == self.__d.len-1:
                    self.__d.phi_rn[s][idx] = self.__w[idx] * 1.;
                else:
                    self.__d.phi_rn[s][idx] = contract(self.__w[idx], self.__d.phi_rn[s][idx+1], False)
                self.__d.phi_rn[s][idx] = contract(self.__d.phi_rn[s][idx], self.__d.phi[s][idx], False)
                self.__d.tl[s].setLabel([-10])
                self.__d.phi_rn[s][idx] = contract(self.__d.phi_rn[s][idx], self.__d.tl[s], False)

    def sweep(self, site_i, site_f, left_to_right=True, normalize=True):
        """"""
        inc = int(left_to_right)*2-1
        for site in xrange(site_i, site_f, inc):
            wi = self.site_tensor_rn(site)
            self.update_w(wi, site, left_to_right, normalize=normalize)
            if site == self.__w.sl:
                self.update_sl_rn(site, left_to_right)
            else:
                self.__d.update_phi_rn(self.__w, site, left_to_right)

    def compress(self, sweeps, left_to_right=True, normalize=True):
        """"""
        forward = left_to_right; backward = not forward
        site_start = 1 if left_to_right else self.__w.px
        site_final = self.__w.px if left_to_right else 1

        if self.__w._ClassifierMPS__bdry_dummy:
            self.__w[0].identity()
            self.__w[-1].identity()

        if self.__w.sl != site_start:
            self.sweep(self.__w.sl, site_start, left_to_right=backward, normalize=normalize)

        for s in xrange(sweeps):
            if s%2 == 0:
                self.sweep(site_start, site_final, left_to_right=forward, normalize=normalize)
            else:
                self.sweep(site_final, site_start, left_to_right=backward, normalize=normalize)

        if self.__w.sl != site_start:
            self.sweep(site_start, self.__w.sl, left_to_right=forward, normalize=normalize)
