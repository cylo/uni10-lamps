import os
from imagemps import *
from utils import tensor_to_ndarray

class ClassifierMPS(MPS):
    """
    """
    def __init__(self, width, height, networks, chi_max=10, dim_label=10, site_with_label_bond=1,
                 dim_virt=1, dim_phys=2, bdry_dummy=True, phys_lab_base=1000):
        """"""
        self.wd = width
        self.ht = height
        self.px = width*height
        MPS.__init__(self, self.px)
        
        self.net = networks
        self.w = self._MPS__mps
        self.chi_max = min(chi_max, 1000) # chi < 1024 to fit pipe limit 64kbit
        self.dv = dim_virt
        self.dp = dim_phys
        self.dl = dim_label
        self.sl = site_with_label_bond-1 # site with label bond. -1 to compensate bdry dummies
        self.assign([Bond(BD_IN, self.dv), Bond(BD_OUT, self.dv), Bond(BD_OUT, self.dp)])
        self.combine_label_bond(true_label=0)
        self.phys_lab_base = phys_lab_base
        self.__bdry_dummy = bdry_dummy
        if bdry_dummy: self.attach_bdry_dummy()
        self.auto_labels()
        
    @classmethod
    def from_image_mps(cls, image_mps, networks, true_label=0, bdry_dummy=True, **kwargs):
        """"""
        clsfy_mps = cls(image_mps.wd, image_mps.ht, networks, phys_lab_base=image_mps.phys_lab_base,
                        dim_virt=1, bdry_dummy=False, **kwargs)
        clsfy_mps.import_tensors(image_mps.phi)
        idn = UniTensor([Bond(BD_IN, 1), Bond(BD_OUT, 1)])
        idn.identity()
        for i in xrange(clsfy_mps.len):
            clsfy_mps.w[i].permute(0)
            clsfy_mps.w[i] = otimes(idn, clsfy_mps.w[i])
        clsfy_mps.combine_label_bond(true_label=true_label)
        if bdry_dummy: clsfy_mps.attach_bdry_dummy()
        clsfy_mps.auto_labels()
        return clsfy_mps
    
    @classmethod
    def from_image_set(cls, image_set, networks, svd=True, bdry_dummy=True, **kwargs):
        """"""
        clsfy_mps = \
            cls.from_image_mps(image_set[0], networks, true_label=image_set[0].label,
                               bdry_dummy=False, **kwargs)
        for i in xrange(1, image_set.size):
            clsfy_mps.direct_sum(
                cls.from_image_mps(image_set[i], networks, true_label=image_set[i].label,
                                   bdry_dummy=False, **kwargs)
            )
        if svd: clsfy_mps.mps_svd()
        if bdry_dummy: clsfy_mps.attach_bdry_dummy()
        clsfy_mps.auto_labels()
        return clsfy_mps
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for i in xrange(self.len):
            self.w[i].save(path + "/W_{}".format(i))
            
    def load(self, path):
        assert os.path.exists(path), 'Path not found.'
        for i in xrange(self.len):
            try:
                self.w[i] = UniTensor(path + "/W_{}".format(i))
            except RuntimeError:
                print "Not enough tensors in {}. Stops at index = {}".format(path, i)
                return
    
    def combine_label_bond(self, true_label, site_with_label_bond=None):
        if site_with_label_bond:
            self.sl = site_with_label_bond
        tl = true_label_vec(true_label, self.dl)
        try:
            self.w[self.sl] = otimes(tl, self.w[self.sl])
        except RuntimeError:
            self.w[self.sl].assign(tl.bond() + self.w[self.sl].bond())
        labs = self.w[self.sl].label()
        self.w[self.sl].permute(labs[1:2]+labs[0:1]+labs[2:], 2)
        self.w[self.sl].setLabel(labs[1:2]+(-10,)+labs[2:])
        
    def attach_bdry_dummy(self):
        dummy = UniTensor([Bond(BD_IN, 1), Bond(BD_OUT, 1)])
        dummy.identity()
        self.insert(0, dummy)
        self.insert(self.len, dummy)
        self.w[0].permute(0)
        self.__bdry_dummy = True
        self.sl += 1
        
    def auto_labels(self, phys_lab_base=None, phys_lab_inc=1):
        if phys_lab_base: self.phys_lab_base = phys_lab_base
        MPS.auto_labels(self, self.phys_lab_base, phys_lab_inc)
        labs_sl = [self.sl, -10, self.sl+1, self.phys_lab_base+phys_lab_inc*self.sl]
        self.w[self.sl].setLabel(labs_sl)
        if self.__bdry_dummy:
            self.w[0].setLabel([1, self.phys_lab_base])
            self.w[self.len-1].setLabel([self.len-1, self.phys_lab_base+phys_lab_inc*(self.len-1)])
        self._MPS__label_ordered = True
        
    def bond_tensor(self, idx=-1, sl_on_left=True):
        """"""
        assert self._MPS__label_ordered, 'Labels of MPS not in order.'
        if idx < 0: idx = self.sl
        if not sl_on_left: idx -= 1
        B = contract(self.w[idx], self.w[idx+1])
        lab_new = [-10] + [x for x in B.label() if x != -10 and x != idx+2] + [idx+2]
        B.permute(lab_new, 1)
        return B
    
    def decision_function(self, img_data, sl_on_left=True, refresh_phi_rn=True):
        """"""
        assert self.__bdry_dummy, 'Boundary dummies not found.'
        site = self.sl
        sl_on_left = sl_on_left and site < self.px
        if refresh_phi_rn: img_data.refresh_phi_rn(self)
        
        idx = site if sl_on_left else site-1
        net = self.net["decision_fn_left"] if sl_on_left else self.net["decision_fn_right"]
        net.putTensor("WI", self.w[idx])
        net.putTensor("WJ", self.w[idx+1])
        net.putTensor("PHIL", img_data.phi_rn[idx-1])
        net.putTensor("PHII", img_data.phi[idx])
        net.putTensor("PHIJ", img_data.phi[idx+1])
        net.putTensor("PHIR", img_data.phi_rn[idx+2])
        df = net.launch()
        return df
    
    def error_function(self, img_data, **kwargs):
        """"""
        ef = img_data.tl + (-1.)*self.decision_function(img_data, **kwargs)
        return ef
    
    def predict(self, img_data, verbose=False, **kwargs):
        """"""
        dfb = self.decision_function(img_data, **kwargs).getBlock()
        df = tensor_to_ndarray(dfb)
        if verbose:
            return df
        return np.argmax(df)
    
    def accuracy(self, image_set, **kwargs):
        """"""
        global USE_MP, PROCS
        correct = []
        for s in xrange(image_set.size):
            correct.append(self.predict(image_set[s], **kwargs) == image_set[s].label)
        return sum(correct)/float(len(correct))
    
    def feature_map(self, label, func=spinor_to_grayscale, snake=False):
        """"""
        assert self.__bdry_dummy, 'Boundary dummies not found.'
        
        # contract the selected label
        lab_vec = true_label_vec(label, self.dl)
        lab_vec.setLabel([-10])
        swl = contract(lab_vec, self.w[self.sl])
        swl.permute(self.w[self.sl].inBondNum()-1)
        
        # define operators for measurement
        N0 = UniTensor([Bond(BD_IN, 2), Bond(BD_OUT, 2)])
        N1 = UniTensor([Bond(BD_IN, 2), Bond(BD_OUT, 2)])
        Id = UniTensor([Bond(BD_IN, 2), Bond(BD_OUT, 2)])
        N0.setElem([1., 0., 0., 0.])
        N1.setElem([0., 0., 0., 1.])
        Id.setElem([1., 0., 0., 1.])
        
        # construct left-vec/right-vec buffer
        lv = [UniTensor() for _ in xrange(self.len)]
        rv = [UniTensor() for _ in xrange(self.len)]
        for i in xrange(self.len-1):
            wi = swl * 1. if i == self.sl else self.w[i] * 1.
            wd = wi * 1.  # w^dagger
            _ , labs = in_out_bonds(wi)
            labs[1][0] *= -1
            if i > 0: labs[0][0] *= -1
            wd.setLabel(labs[0]+labs[1])
            lv[i] = wi * wd if i == 0 else wi * lv[i-1] * wd
        for i in xrange(self.len-1, 0, -1):
            wi = swl * 1. if i == self.sl else self.w[i] * 1.
            wd = wi * 1.
            _ , labs = in_out_bonds(wi)
            labs[0][0] *= -1
            if i < self.len-1: labs[1][0] *= -1
            wd.setLabel(labs[0]+labs[1])
            rv[i] = wi * wd if i == self.len-1 else wi * rv[i+1] * wd
        
        # measurement
        res = []; norm = 1.
        for i in xrange(1, self.px+1):
            wi = swl * 1. if i == self.sl else self.w[i] * 1.
            wd = wi * 1.
            net = self.net["measurement"]
            net.putTensor("WI", wi)
            net.putTensor("WD", wd)
            net.putTensor("LV", lv[i-1])
            net.putTensor("RV", rv[i+1])
            net.putTensor("OP", N0)
            n0 = net.launch()[0]
            net.putTensor("OP", N1)
            n1 = net.launch()[0]
            if i == 1:
                net.putTensor("OP", Id)
                norm = net.launch()[0]
            phi = UniTensor([Bond(BD_IN, 2)])
            phi.setElem([n0/norm, n1/norm])
            res += [phi]
            
        # convert mps into image
        img_mps = ImageMPS(np.zeros((self.wd, self.ht)))
        img_mps.import_tensors(res)
        if snake:
            return img_mps.mps_to_img_snake(func)
        return img_mps.mps_to_img(func)
