import os
import multiprocessing as mp
from itertools import izip

from mps.classifiermps import *
from utils import l2_norm
import shared

class GDOptimizer(object):
    """
    """
    def __init__(self, image_set, clsfy_mps, networks, net_dir=shared.NETDIR+"/gdo", step_size=-1., mproc=None):
        """"""
        self.__d = image_set
        self.__w = clsfy_mps
        self.step = step_size if step_size > 0 else (1./float(self.__d.size))
        self.net = networks
        self.net_dir = net_dir
        self.load_networks(net_dir)
        self.__d.refresh_phi_rn(self.__w)
        if mproc:
            shared.USE_MP = True
            shared.PROCS = int(mproc)

    def load_networks(self, net_dir=None):
        """"""
        if net_dir:
            self.net_dir = net_dir
        self.net["label_projection"] = Network(self.net_dir + "/label_projection.net")

    def label_projection(self, label_vec, img_data, site=-1, sl_on_left=True, normalize_phi_rn=True):
        """"""
        if site < 0: site = self.__w.sl
        idx = site if sl_on_left else site-1
        scale = [1., 1., 1., 1.]
        if normalize_phi_rn:
            scale[0] = (1./l2_norm(img_data.phi_rn[idx-1]))
            scale[1] = (1./l2_norm(img_data.phi[idx]))
            scale[2] = (1./l2_norm(img_data.phi[idx+1]))
            scale[3] = (1./l2_norm(img_data.phi_rn[idx+2]))
        net = self.net["label_projection"]
        net.putTensor("LAB", label_vec)
        net.putTensor("PHIL", img_data.phi_rn[idx-1] * scale[0])
        net.putTensor("PHII", img_data.phi[idx] * scale[1])
        net.putTensor("PHIJ", img_data.phi[idx+1] * scale[2])
        net.putTensor("PHIR", img_data.phi_rn[idx+2] * scale[3])
        proj = net.launch()
        return proj
    
    def bond_tensor_grad(self, site=-1, sl_on_left=True, normalize_phi_rn=True, **kwargs):
        """"""
        dB = UniTensor(); dBn = UniTensor()
        if site < 0: site = self.__w.sl
        if site != self.__w.sl:
            raise IndexError('Bond tensor index not matched.')
        
        if shared.USE_MP:
            shared.W = self.__w.w
            shared.PHI = self.__d.phi
            shared.PHI_RN = self.__d.phi_rn
            shared.TL = self.__d.tl
            shared.NET = self.net
            
            os.environ["OMP_NUM_THREADS"] = "1"
            batch = (self.__d.size)//shared.PROCS
            pool = mp.Pool(shared.PROCS)
            res = pool.map(mp_ef_projection,
                           izip([batch*i for i in xrange(shared.PROCS)],
                                [batch*(i+1) for i in xrange(shared.PROCS-1)]+[self.__d.size],
                                [site for _ in xrange(shared.PROCS)],
                                [sl_on_left for _ in xrange(shared.PROCS)],
                                [normalize_phi_rn for _ in xrange(shared.PROCS)]))
            elems = sum(res)
            dB = self.__w.bond_tensor(site, sl_on_left)
            dB.setElemR(elems)
            pool.close()
            os.environ["OMP_NUM_THREADS"] = str(shared.PROCS)
            
        else:
            for img_data in self.__d:
                ef = self.__w.error_function(img_data, sl_on_left=sl_on_left, **kwargs)
                dBn = self.label_projection(ef, img_data, sl_on_left=sl_on_left, normalize_phi_rn=normalize_phi_rn)
                try:
                    dB += dBn
                except:
                    dB = dBn * 1.
        return dB
    
    def update_w(self, bond_tensor, site=-1, left_to_right=True, cutoff=1e-10, normalize=True):
        """"""
        if site < 0: site = self.__w.sl    
        idx = site if left_to_right else site-1
        th_orig = contract(self.__w[idx], self.__w[idx+1])
        _ , labo = in_out_bonds(th_orig)
        labth = [l for l in labo[0] if l != -10] + [-10] + labo[1] if left_to_right \
            else labo[0] + [-10] + [l for l in labo[1] if l != -10]
        th_struct = (labth, len(labo[1]))
        
        bond_svd(self.__w[idx], self.__w[idx+1], self.__w.chi_max, bond_tensor, th_struct,
                 merge_sv_right=left_to_right, cutoff=cutoff, normalize=normalize, perm_back=False)
        
        lab1 = self.__w[idx].label()
        if left_to_right:
            self.__w[idx].permute([lab1[0],lab1[2],lab1[1]], 1)
            self.__w[idx+1].permute(2)
            self.__w.sl += 1
        else:
            self.__w[idx].permute([lab1[0],lab1[2],lab1[3],lab1[1]], 2)
            self.__w[idx+1].permute(1)
            self.__w.sl -= 1
    
    def sweep(self, site_i, site_f, step_size=-1, left_to_right=True, normalize=True):
        """"""
        if step_size > 0: self.step = step_size
        inc = int(left_to_right)*2-1
        for site in xrange(site_i, site_f, inc):
            B = self.__w.bond_tensor(site, left_to_right)
            dB = self.bond_tensor_grad(site, sl_on_left=left_to_right,
                                       normalize_phi_rn=normalize, refresh_phi_rn=False)
            B += self.step * dB
            self.update_w(B, site, left_to_right, normalize=normalize)
            self.__d.update_phi_rn(self.__w, site, left_to_right)
            
    def gradient_descent(self, sweeps, site_term=1, step_size=-1, left_to_right=True, normalize=True):
        """"""
        forward = left_to_right; backward = not forward
        site_start = 1 if left_to_right else self.__w.px
        site_final = self.__w.px if left_to_right else 2
        inc = int(left_to_right)*2-1
        
        if self.__w._ClassifierMPS__bdry_dummy:
            self.__w[0].identity()
            self.__w[-1].identity()

        if self.__w.sl != site_start:
            self.sweep(self.__w.sl, site_start,
                       step_size=step_size, left_to_right=backward, normalize=normalize)
        for s in xrange(sweeps):
            if s%2 == 0:
                self.sweep(site_start, site_final,
                           step_size=step_size, left_to_right=forward, normalize=normalize)
            else:
                self.sweep(site_final, site_start,
                           step_size=step_size, left_to_right=backward, normalize=normalize)
        if self.__w.sl != site_term:
            self.sweep(self.__w.sl, site_term,
                       step_size=step_size, left_to_right=(forward and (sweeps-1)%2), normalize=normalize)


def mp_ef_projection(args):
    """"""
    sample_start, sample_end, site, sl_on_left, normalize_phi_rn = args
    idx = site if sl_on_left else site-1
    netd = shared.NET["decision_fn_left"] if sl_on_left else shared.NET["decision_fn_right"]
    netp = shared.NET["label_projection"]
    dB = UniTensor()
    
    for s in xrange(sample_start, sample_end):
        netd.putTensor("WI", shared.W[idx])
        netd.putTensor("WJ", shared.W[idx+1])
        netd.putTensor("PHIL", shared.PHI_RN[s][idx-1])
        netd.putTensor("PHII", shared.PHI[s][idx])
        netd.putTensor("PHIJ", shared.PHI[s][idx+1])
        netd.putTensor("PHIR", shared.PHI_RN[s][idx+2])
        df = netd.launch()
        # df.permute(1) # Mysteriously, sometimes Pool.map mess up the final permute in launch!
        ef = shared.TL[s] + (-1.)*df

        scale = [1., 1., 1., 1.]
        if normalize_phi_rn:
            scale[0] = (1./l2_norm(shared.PHI_RN[s][idx-1]))
            scale[1] = (1./l2_norm(shared.PHI[s][idx]))
            scale[2] = (1./l2_norm(shared.PHI[s][idx+1]))
            scale[3] = (1./l2_norm(shared.PHI_RN[s][idx+2]))
        netp.putTensor("LAB", ef)
        netp.putTensor("PHIL", shared.PHI_RN[s][idx-1] * scale[0])
        netp.putTensor("PHII", shared.PHI[s][idx] * scale[1])
        netp.putTensor("PHIJ", shared.PHI[s][idx+1] * scale[2])
        netp.putTensor("PHIR", shared.PHI_RN[s][idx+2] * scale[3])
        proj = netp.launch()
        try:
            dB += proj
        except:
            dB = proj
            
    return exportElem(dB)
