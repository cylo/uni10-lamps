import os
import multiprocessing as mp
from itertools import izip

from mps.classifiermps import *
from utils import l2_norm, matrix_to_ndarray
import shared

class FixedGDOptimizer(object):
    """
    """
    def __init__(self, image_set, clsfy_mps, networks, net_dir=shared.NETDIR+"/fixed-gdo", step_size=-1., mproc=None):
        """"""
        self.__d = image_set
        self.__w = clsfy_mps
        self.step = step_size if step_size > 0 else (1./float(self.__d.size))
        self.net = networks
        self.net_dir = net_dir
        self.load_networks(net_dir)
        self.__d.refresh_phi_rn(self.__w)
        if mproc:
            shared.OLD_VER = False
            shared.USE_MP = True
            shared.PROCS = int(mproc)

    def load_networks(self, net_dir=None):
        """"""
        if net_dir:
            self.net_dir = net_dir
        self.net["bt_project_sll"] = Network(self.net_dir + "/bt_project_sll.net")
        self.net["bt_project_slr"] = Network(self.net_dir + "/bt_project_slr.net")
        self.net["decision_fn_sll"] = Network(self.net_dir + "/decision_fn_sll.net")
        self.net["decision_fn_slr"] = Network(self.net_dir + "/decision_fn_slr.net")
        self.net["label_projection"] = Network(self.net_dir + "/label_projection.net")

    @staticmethod
    def decision_function(self, img_data, site=-1, left_to_right=True):
        """"""
        ## self = ClassifierMPS
        if site < 0: site = img_data.rnb
        idx = site if left_to_right else site-1
        sl_in_left_rn = bool(idx > self.sl)
        sl_in_right_rn = bool(idx+1 < self.sl)
        
        if sl_in_left_rn:
            net = self.net["decision_fn_sll"]
        elif sl_in_right_rn:
            net = self.net["decision_fn_slr"]
        else:
            net = self.net["decision_fn_left"] if idx == self.sl else self.net["decision_fn_right"]
        net.putTensor("WI", self.w[idx])
        net.putTensor("WJ", self.w[idx+1])
        net.putTensor("PHIL", img_data.phi_rn[idx-1])
        net.putTensor("PHII", img_data.phi[idx])
        net.putTensor("PHIJ", img_data.phi[idx+1])
        net.putTensor("PHIR", img_data.phi_rn[idx+2])
        df = net.launch()
        return df

    def label_projection(self, label_vec, img_data, site=-1, left_to_right=True, normalize_phi_rn=True):
        """"""
        if site < 0: site = img_data.rnb
        idx = site if left_to_right else site-1
        sl_in_left_rn = bool(idx > self.__w.sl)
        sl_in_right_rn = bool(idx+1 < self.__w.sl)
        scale = [1., 1., 1., 1.]
        if normalize_phi_rn:
            scale[0] = inv_phi_rn(img_data.phi_rn[idx-1]) if sl_in_left_rn else (1./l2_norm(img_data.phi_rn[idx-1]))
            scale[1] = (1./l2_norm(img_data.phi[idx]))
            scale[2] = (1./l2_norm(img_data.phi[idx+1]))
            scale[3] = inv_phi_rn(img_data.phi_rn[idx+2]) if sl_in_right_rn else (1./l2_norm(img_data.phi_rn[idx+2]))
        if sl_in_left_rn:
            net = self.net["bt_project_sll"]
            scale[0].transpose()
            net.putTensor("LAB", label_vec)
            net.putTensor("PHIL", scale[0])
            net.putTensor("PHII", img_data.phi[idx] * scale[1])
            net.putTensor("PHIJ", img_data.phi[idx+1] * scale[2])
            net.putTensor("PHIR", img_data.phi_rn[idx+2] * scale[3])
        elif sl_in_right_rn:
            net = self.net["bt_project_slr"]
            scale[3].transpose()
            net.putTensor("LAB", label_vec)
            net.putTensor("PHIL", img_data.phi_rn[idx-1] * scale[0])
            net.putTensor("PHII", img_data.phi[idx] * scale[1])
            net.putTensor("PHIJ", img_data.phi[idx+1] * scale[2])
            net.putTensor("PHIR", scale[3])
        else:
            net = self.net["label_projection"]
            net.putTensor("LAB", label_vec)
            net.putTensor("PHIL", img_data.phi_rn[idx-1] * scale[0])
            net.putTensor("PHII", img_data.phi[idx] * scale[1])
            net.putTensor("PHIJ", img_data.phi[idx+1] * scale[2])
            net.putTensor("PHIR", img_data.phi_rn[idx+2] * scale[3])
        proj = net.launch()
        return proj

    def bond_tensor(self, site, left_to_right):
        """"""
        assert self.__w._MPS__label_ordered, 'Labels of MPS not in order.'
        idx = site if left_to_right else site-1
        B = contract(self.__w[idx], self.__w[idx+1])
        lab_b = B.label()
        if -10 in lab_b:
            lab_new = [-10] + [x for x in lab_b if x != -10 and x != idx+2] + [idx+2]
            B.permute(lab_new, 1)
        else:
            lab_new = [x for x in lab_b if x != idx+2] + [idx+2]
            B.permute(lab_new, 0)
        return B

    def bond_tensor_grad(self, site=-1, left_to_right=True, normalize_phi_rn=True, **kwargs):
        """"""
        dB = UniTensor(); dBn = UniTensor()
        if site < 0: site = self.__d[0].rnb
        
        if shared.USE_MP:
            shared.W = self.__w.w
            shared.PHI = self.__d.phi
            shared.PHI_RN = self.__d.phi_rn
            shared.TL = self.__d.tl
            shared.NET = self.net
            
            os.environ["OMP_NUM_THREADS"] = "1"
            batch = (self.__d.size)//shared.PROCS
            pool = mp.Pool(shared.PROCS)
            res = pool.map(mp_fgd_ef_project,
                           izip([batch*i for i in xrange(shared.PROCS)],
                                [batch*(i+1) for i in xrange(shared.PROCS-1)]+[self.__d.size],
                                [site for _ in xrange(shared.PROCS)],
                                [self.__w.sl for _ in xrange(shared.PROCS)],
                                [left_to_right for _ in xrange(shared.PROCS)],
                                [normalize_phi_rn for _ in xrange(shared.PROCS)]))
            elems = sum(res)
            dB = self.bond_tensor(site, left_to_right)
            dB.setElemR(elems)
            pool.close()
            os.environ["OMP_NUM_THREADS"] = str(shared.PROCS)
        else:
            for img_data in self.__d:
                ef = self.__w.error_function(img_data, dcsfn=FixedGDOptimizer.decision_function,
                                             site=site, left_to_right=left_to_right)
                dBn = self.label_projection(ef, img_data, site, left_to_right, normalize_phi_rn=normalize_phi_rn)
                try:
                    dB += dBn
                except:
                    dB = dBn * 1.
        return dB

    def update_w(self, bond_tensor, site, left_to_right=True, cutoff=0., normalize=True, **kwargs):
        """"""
        idx = site if left_to_right else site-1
        lab1 = self.__w[idx].label()
        lab2 = self.__w[idx+1].label()
        ibn1 = self.__w[idx].inBondNum()
        ibn2 = self.__w[idx+1].inBondNum()
        th_orig = contract(self.__w[idx], self.__w[idx+1])
        th_struct = (th_orig.label(), th_orig.inBondNum())
        
        bond_svd(self.__w[idx], self.__w[idx+1], self.__w.chi_max, bond_tensor, th_struct,
                 merge_sv_right=left_to_right, cutoff=cutoff, normalize=normalize, perm_back=False, **kwargs)

        self.__w[idx].permute(lab1, ibn1)
        self.__w[idx+1].permute(lab2, ibn2)

    def sweep(self, site_i, site_f, step_size=-1, left_to_right=True, normalize=True):
        """"""
        if step_size > 0: self.step = step_size
        inc = int(left_to_right)*2-1
        for site in xrange(site_i, site_f, inc):
            B = self.bond_tensor(site, left_to_right)
            dB = self.bond_tensor_grad(site, left_to_right=left_to_right, normalize_phi_rn=normalize)
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
        if self.__d[0].rnb != self.__w.sl:
            self.sweep(self.__d[0].rnb, self.__w.sl,
                       step_size=step_size, left_to_right=(forward and (sweeps-1)%2), normalize=normalize)


def mp_fgd_ef_project(args):
    """"""
    sample_start, sample_end, site, sl_idx, left_to_right, normalize_phi_rn = args
    idx = site if left_to_right else site-1
    sl_in_left_rn = bool(idx > sl_idx)
    sl_in_right_rn = bool(idx+1 < sl_idx)
    if sl_in_left_rn:
        netd = shared.NET["decision_fn_sll"]
        netp = shared.NET["bt_project_sll"]
    elif sl_in_right_rn:
        netd = shared.NET["decision_fn_slr"]
        netp = shared.NET["bt_project_slr"]
    else:
        netd = shared.NET["decision_fn_left"] if idx == sl_idx else shared.NET["decision_fn_right"]
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
            scale[0] = inv_phi_rn(shared.PHI_RN[s][idx-1]) if sl_in_left_rn else (1./l2_norm(shared.PHI_RN[s][idx-1]))
            scale[1] = (1./l2_norm(shared.PHI[s][idx]))
            scale[2] = (1./l2_norm(shared.PHI[s][idx+1]))
            scale[3] = inv_phi_rn(shared.PHI_RN[s][idx+2]) if sl_in_right_rn else (1./l2_norm(shared.PHI_RN[s][idx+2]))
        if sl_in_left_rn:
            scale[0].transpose()
            netp.putTensor("LAB", ef)
            netp.putTensor("PHIL", scale[0])
            netp.putTensor("PHII", shared.PHI[s][idx] * scale[1])
            netp.putTensor("PHIJ", shared.PHI[s][idx+1] * scale[2])
            netp.putTensor("PHIR", shared.PHI_RN[s][idx+2] * scale[3])
        elif sl_in_right_rn:
            scale[3].transpose()
            netp.putTensor("LAB", ef)
            netp.putTensor("PHIL", shared.PHI_RN[s][idx-1] * scale[0])
            netp.putTensor("PHII", shared.PHI[s][idx] * scale[1])
            netp.putTensor("PHIJ", shared.PHI[s][idx+1] * scale[2])
            netp.putTensor("PHIR", scale[3])
        else:
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


def inv_phi_rn(phi_rn, old_ver=shared.OLD_VER):
    """"""
    assert phi_rn.bondNum() == 2, "phi_rn's bond number != 2."
    inv = phi_rn * 1.
    inv.setLabel([0, 1])
    inv.permute(1)
    if old_ver:
        phi_blk = inv.getBlock()
        inv_blk = phi_blk * 1
        inv_blk.transpose()
        inv_blk.setElem((np.linalg.pinv(matrix_to_ndarray(phi_blk))).ravel())
        inv.permute([1, 0], 1)
        inv.putBlock(inv_blk)
    else:
        phi_blk = inv.getBlockNparray()
        inv.permute([1, 0], 1)
        inv.putBlockNparray(np.linalg.pinv(phi_blk))
    return inv
