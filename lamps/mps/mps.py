from pyUni10 import *
import numpy as np

from utils import matrix_to_ndarray, tensor_to_ndarray

class MPS(object):
    """
    """
    def __init__(self, len_mps=1, chi_max=10):
        self.__mps = [UniTensor() for i in xrange(len_mps)]
        self.len = len_mps
        self.chi_max = chi_max
        self.obc = True
        self.__label_ordered = False
        
    @classmethod
    def from_mps(cls, other):
        mps = cls(other.len, other.chi_max)
        mps.import_tensors(other)
        return mps
    
    @classmethod
    def from_list(cls, list_ten, chi_max=10):
        len_mps = len(list_ten)
        mps = cls(len_mps, chi_max)
        mps.import_tensors(list_ten)
        return mps
    
    def import_tensors(self, tensors):
        for i in xrange(self.len):
            self.__mps[i] = tensors[i] * 1.

    def __getitem__(self, idx):
        return self.__mps[idx]
    
    def __setitem__(self, idx, val):
        self.__mps[idx] = val
        
    def assign(self, bonds):
        for i in xrange(self.len):
            self.__mps[i].assign(bonds)
        if self.obc and not self.is_product():
            self.assign_obc()
            
    def assign_obc(self, obc_in_idx=0, obc_out_idx=0):
        # stick virt bonds to 1st in-bond and 1st out-bond
        bdl, _ = in_out_bonds(self.__mps[0])
        bdr, _ = in_out_bonds(self.__mps[-1])
        bdl[0][obc_in_idx] = Bond(BD_IN, 1)
        bdr[1][obc_out_idx] = Bond(BD_OUT, 1)
        self.__mps[0].assign(bdl[0] + bdl[1])
        self.__mps[-1].assign(bdr[0] + bdr[1])
        self.obc = True
            
    def randomize(self, scale=1.):
        for i in xrange(self.len):
            self.__mps[i].randomize()
            self.__mps[i] *= scale
            
    def uniform(self, scale=1.):
        for i in xrange(self.len):
            elem = np.ones(self.__mps[i].elemNum()) * scale
            self.__mps[i].setElem(elem)
            
    def insert(self, idx, ut):
        self.__mps.insert(idx, ut*1.)
        self.len += 1
        self.__label_ordered = False
            
    def is_product(self):
        """check product state"""
        ibn = self.__mps[-1].inBondNum()
        obn = self.__mps[0].bondNum() - self.__mps[0].inBondNum()
        return (ibn == 0 or obn == 0)
            
    def direct_sum(self, other, sum_chi=False):
        prod1 = self.is_product()
        prod2 = other.is_product()
        if self.obc or other.obc:
            self.__mps[0] = concat(self.__mps[0], other[0], True, prod1, prod2)
            self.__mps[-1] = concat(self.__mps[-1], other[-1], False, prod1, prod2)
            for i in xrange(1, self.len-1):
                self.__mps[i] = direct_sum(self.__mps[i], other[i], prod1, prod2)
        else:
            for i in xrange(self.len):
                self.__mps[i] = direct_sum(self.__mps[i], other[i], prod1, prod2)
        if sum_chi:
            self.chi_max += other.chi_max
        self.__label_ordered = False
            
    def auto_labels(self, phys_lab_base=10000, phys_lab_inc=10, phys_lab_inc_onsite=1):
        prod = self.is_product()
        for i in xrange(self.len):
            bd, _ = in_out_bonds(self.__mps[i])
            len_bdi = len(bd[0])
            len_bdo = len(bd[1])
            len_bds = len_bdi+len_bdo
            labs = range(len_bds)
            if prod:
                for n in xrange(len_bds):
                    labs[n] = phys_lab_base + phys_lab_inc*i + phys_lab_inc_onsite*(n)
            else:
                for n in xrange(len_bds):
                    if n == 0: labs[n] = i
                    elif n == len_bdi: labs[n] = i+1
                    else:
                        m = n-1 if n < len_bdi else n-2
                        labs[n] = phys_lab_base + phys_lab_inc*i + phys_lab_inc_onsite*(m)
                    
            self.__mps[i].setLabel(labs)
        self.__label_ordered = True
        
    def mps_svd(self, chi_max=-1, sweep_left_to_right=False, cutoff=1e-10, normalize=True, **kwargs):
        if chi_max > 0: self.chi_max = chi_max
        if not self.__label_ordered: self.auto_labels()
        loop_range = xrange(self.len-1) if sweep_left_to_right else xrange(self.len-2, -1, -1)
        for i in loop_range:
            bond_svd(self.__mps[i], self.__mps[i+1], self.chi_max,
                     merge_sv_right=sweep_left_to_right, cutoff=cutoff, normalize=normalize, **kwargs)
            
    def add(self, other, chi_max=-1, cutoff=1e-10, normalize=True, **kwargs):
        self.direct_sum(other, False)
        self.chi_max = chi_max if chi_max > 0 else max(self.chi_max, other.chi_max)
        self.mps_svd(self.chi_max, False, cutoff, normalize, **kwargs)
            
    def __add__(self, other):
        mps = MPS.from_mps(self)
        mps.add(other)
        return mps
    
    def __iadd__(self, other):
        self.add(other)
        return self
    
    def __mul__(self, scale):
        mps = MPS.from_mps(self)
        for i in xrange(mps.len):
            mps[i] *= scale
        return mps
    
    def __imul__(self, scale):
        for i in xrange(self.len):
            self.__mps[i] *= scale
        return self
    
    def __len__(self):
        return self.len
    
    def __str__(self):
        assert self.len == len(self.__mps), 'MPS length settings inconsistent.'
        if self.len > 0:
            is_prod = self.is_product()
            ut_str = UniTensor(self.__mps[self.len/2].bond()).__str__()
        else:
            is_prod = None
            ut_str = None
        return ("MPS of  length = {}\n\t" +
                "bond-dim upper limit (for svd) = {}\n\t" +
                "open-boundary-condition = {}\n\t" +
                "label-ordered = {}\n\t" +
                "product-state = {}\n\t" +
                "example tensor structure:\n{}"
               ).format(self.len, self.chi_max, self.obc, self.__label_ordered, is_prod, ut_str) 


def direct_sum_ndarray(nda1, nda2):
    dsum = np.zeros( np.add(nda1.shape, nda2.shape) )
    dsum[:nda1.shape[0],:nda1.shape[1]] = nda1
    dsum[nda1.shape[0]:,nda1.shape[1]:] = nda2
    return dsum

def concat_ndarray(nda1, nda2, axis=0):
    return np.concatenate((nda1, nda2), axis=axis)

def in_out_bonds(ut):
    bds = ut.bond()
    len_bds = len(bds)
    bdi = [bds[i] for i in xrange(len_bds) if bds[i].type() == 1]
    bdo = [bds[i] for i in xrange(len_bds) if bds[i].type() == -1]
    labs = ut.label()
    labi = [labs[i] for i in xrange(len_bds) if bds[i].type() == 1]
    labo = [labs[i] for i in xrange(len_bds) if bds[i].type() == -1]
    return (bdi, bdo), (labi, labo)

def summable(bd1, bd2, prod1, prod2):
    bn1 = len(bd1[0]+bd1[1])
    bn2 = len(bd2[0]+bd2[1])
    bond_num_ne = (bn1 != bn2) and (prod1 == prod2)
    bdi_type_ne = (bd1[0][int(not prod1):] != bd2[0][int(not prod2):])
    bdo_type_ne = (bd1[1][int(not prod1):] != bd2[1][int(not prod2):])
    if bond_num_ne or bdi_type_ne or bdo_type_ne:
        return False
    return True

def direct_sum(ut1, ut2, prod1=True, prod2=True):
    bd1, _ = in_out_bonds(ut1)
    bd2, _ = in_out_bonds(ut2)
    if not summable(bd1, bd2, prod1, prod2):
        raise RuntimeError("Cannot perform direct sum of two tensors having diffirent bond types.")
        
    if prod1:
        bd1[0].insert(0, Bond(BD_IN, 1))
        bd1[1].insert(0, Bond(BD_OUT, 1))
    if prod2:
        bd2[0].insert(0, Bond(BD_IN, 1))
        bd2[1].insert(0, Bond(BD_OUT, 1))
    bd1[0][0] = Bond(BD_IN, bd1[0][0].dim()+bd2[0][0].dim())
    bd1[1][0] = Bond(BD_OUT, bd1[1][0].dim()+bd2[1][0].dim())
    bonds = bd1[0] + bd1[1]
    
    ndsum = direct_sum_ndarray(matrix_to_ndarray(ut1.getBlock()), matrix_to_ndarray(ut2.getBlock()))
    dsum = UniTensor(bonds)
    dsum.setElem(np.reshape(ndsum, ndsum.size))
    return dsum

def concat(ut1, ut2, left_bdry=True, prod1=True, prod2=True):
    bd1, _ = in_out_bonds(ut1)
    bd2, _ = in_out_bonds(ut2)
    if not summable(bd1, bd2, prod1, prod2):
        raise RuntimeError("Cannot perform direct sum of two tensors having diffirent bond types.")
        
    if prod1:
        bd1[0].insert(0, Bond(BD_IN, 1))
        bd1[1].insert(0, Bond(BD_OUT, 1))
    if prod2:
        bd2[0].insert(0, Bond(BD_IN, 1))
        bd2[1].insert(0, Bond(BD_OUT, 1))
    if left_bdry:
        bd1[0][0] = Bond(BD_IN, 1)
        bd1[1][0] = Bond(BD_OUT, bd1[1][0].dim()+bd2[1][0].dim())
        axis = 1
    else:
        bd1[0][0] = Bond(BD_IN, bd1[0][0].dim()+bd2[0][0].dim())
        bd1[1][0] = Bond(BD_OUT, 1)
        axis = 0
    bonds = bd1[0] + bd1[1]
    
    ndsum = concat_ndarray(matrix_to_ndarray(ut1.getBlock()), matrix_to_ndarray(ut2.getBlock()), axis)
    dsum = UniTensor(bonds)
    dsum.setElem(np.reshape(ndsum, ndsum.size))
    return dsum

def common_label(lab1, lab2):
    return tuple(set(lab1).intersection(lab2))

def cutoff_dim(diag_mtx, cutoff):
    if not diag_mtx.isDiag():
        raise ValueError("Matrix is not diagonal.")
    chi = diag_mtx.col()
    for i in xrange(chi):
        if diag_mtx[i] < cutoff:
            return i
    return chi

def bond_svd(ut1, ut2, chi_max, theta=None, th_struct=(None, None),
             merge_sv_right=True, cutoff=1e-10, normalize=True, perm_back=True, show_sv=False):
    ibn1 = ut1.inBondNum()
    ibn2 = ut2.inBondNum()
    lab1 = ut1.label()
    lab2 = ut2.label()
    common = common_label(lab1, lab2)
    if len(common) != 1:
        raise RuntimeError("Cannot perform SVD update other than single bond.")
    
    if theta == None:
        theta = contract(ut1, ut2)
    if th_struct[0] and th_struct[1]:
        theta.permute(th_struct[0], th_struct[1])
    svd = theta.getBlock().svd()
    chi = min(chi_max, cutoff_dim(svd[1], cutoff))
            
    svd[0].resize(svd[0].row(), chi)
    svd[1].resize(chi, chi)
    sv_norm = svd[1].norm()
    if 0 < float(normalize) <= 1:
        sv = svd[1] * (1./sv_norm)
    elif 1 < float(normalize) < sv_norm:
        sv = svd[1] * (np.sqrt(float(normalize))/sv_norm)
    else:
        sv = svd[1]
    svd[2].resize(chi, svd[2].col())
    if show_sv:
        print exportElem(sv), sv_norm

    bdth, labth = in_out_bonds(theta)
    bd1_new = bdth[0] + [Bond(BD_OUT, chi)]
    bd2_new = [Bond(BD_IN, chi)] + bdth[1]
    ut1.assign(bd1_new)
    ut2.assign(bd2_new)
    if merge_sv_right:
        ut1.putBlock(svd[0])
        ut2.putBlock(sv * svd[2])
    else:
        ut1.putBlock(svd[0] * sv)
        ut2.putBlock(svd[2])
    ut1.setLabel(labth[0] + [common[0]])
    ut2.setLabel([common[0]] + labth[1])
    if perm_back:
        ut1.permute(lab1, ibn1)
        ut2.permute(lab2, ibn2)
