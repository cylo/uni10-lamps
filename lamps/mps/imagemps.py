import os

from preprocessing.pixelconverter import grayscale_to_spinor, spinor_to_grayscale
from utils import true_label_vec
# import shared
from mps import *

class ImageMPS(MPS):
    """
    """
    def __init__(self, image_nda=np.zeros((1,1)), func=grayscale_to_spinor,
                 bdry_dummy=False, snake=False, phys_lab_base=1000):
        """
        Construct Image MPS from ndarray and a converson function
        """
        MPS.__init__(self, image_nda.size)
        self.img = image_nda
        self.convert_func = func
        self.phys_lab_base = phys_lab_base
        self.phi = self._MPS__mps
        self.__bdry_dummy = bdry_dummy
        self.__update_info()
        
        self.dp = len(self.convert_func(self.img[0][0]))  # phys dim
        self.assign([Bond(BD_IN, self.dp)])
        if snake:
            self.__img_to_mps_snake()
        else:
            self.__img_to_mps()
        if bdry_dummy:
            self.attach_bdry_dummy()
        self.auto_labels(self.phys_lab_base, phys_lab_inc=1)
        
    def __update_info(self):
        """"""
        self.wd = self.img.shape[0]
        self.ht = self.img.shape[1]
        self.px = self.img.size
        
    def __img_to_mps(self):
        """
        Build MPS for image
        """
        for i in xrange(self.wd):
            for j in xrange(self.ht):
                self.phi[i*self.wd + j].setElem(self.convert_func(self.img[i][j]))
                
    def __img_to_mps_snake(self):
        """
        Build MPS for image in snake ordering
        """
        for i in xrange(self.wd):
            for j in xrange(self.ht):
                if i%2 == 0:
                    self.phi[i*self.wd + j].setElem(self.convert_func(self.img[i][j]))
                else:
                    self.phi[i*self.wd + j].setElem(self.convert_func(self.img[i][self.ht-1-j]))
                    
    def attach_bdry_dummy(self):
        dummy = UniTensor([Bond(BD_IN, 1)])
        dummy.identity()
        self.insert(0, dummy)
        self.insert(self.len, dummy)
        
    def mps_to_img(self, func=spinor_to_grayscale):
        """"""
        loop_range = xrange(1, self.px+1) if self.__bdry_dummy else xrange(self.len)
        nda = np.array([func(self.phi[i]) for i in loop_range])
        nda = nda.reshape((self.wd, self.ht))
        return nda
    
    def mps_to_img_snake(self, func=spinor_to_grayscale):
        """"""
        loop_range = xrange(1, self.px+1) if self.__bdry_dummy else xrange(self.len)
        colors = [0. for _ in loop_range]
        for i in xrange(self.wd):
            for j in xrange(self.ht):
                if i%2 == 0:
                    colors[i*self.wd + j] = func(self.phi[i*self.wd + j])
                else:
                    colors[i*self.wd + j] = func(self.phi[i*self.wd + self.ht-1-j])
        nda = np.array(colors)
        nda = nda.reshape((self.wd, self.ht))
        return nda


class ImageData(ImageMPS):
    """
    """
    def __init__(self, image_mps=None, true_label=None, dim_l=10, **kwargs):
        """"""
        if image_mps:
            self.__dict__ = dict(image_mps.__dict__)  # full reference to image_mps
        else:
            ImageMPS.__init__(self, **kwargs)
        self.label = true_label
        self.dl = dim_l
        self.phi_rn = [UniTensor() for _ in xrange(self.len)]
        self.tl = true_label_vec(self.label, self.dl) if true_label != None else None
        self.rnb = None  # boundary of left/right/custom phi_rn

    def refresh_phi_rn(self, clsfy_mps):
        site_with_label_bond = clsfy_mps.sl
        for i in xrange(self.len):
            self.phi_rn[i] = UniTensor()
        for i in xrange(site_with_label_bond):
            self.renorm_phi_left(clsfy_mps, i)
        for i in xrange(self.len-1, site_with_label_bond, -1):
            self.renorm_phi_right(clsfy_mps, i)

    def renorm_phi_left(self, clsfy_mps, idx):
        """"""
        if idx == 0:
            self.phi_rn[idx] = clsfy_mps[idx] * 1.;
        else:
            self.phi_rn[idx] = contract(self.phi_rn[idx-1], clsfy_mps[idx], False)
        self.phi_rn[idx] = contract(self.phi_rn[idx], self.phi[idx], False)
        if self.phi_rn[idx].inBondNum() > 1: self.phi_rn[idx].permute(1)
        self.rnb = max(self.rnb, idx+1)
        
    def renorm_phi_right(self, clsfy_mps, idx):
        """"""
        if idx == self.len-1:
            self.phi_rn[idx] = clsfy_mps[idx] * 1.;
        else:
            self.phi_rn[idx] = contract(clsfy_mps[idx], self.phi_rn[idx+1], False)
        self.phi_rn[idx] = contract(self.phi_rn[idx], self.phi[idx], False)
        if self.phi_rn[idx].inBondNum() > 1: self.phi_rn[idx].permute(1)
        self.rnb = idx-1 if self.rnb == None else min(self.rnb, idx-1)

    def refresh_rn_custom(self, clsfy_mps, functors=[], mode=0, **kwargs):
        site_with_label_bond = clsfy_mps.sl
        for i in xrange(self.len):
            self.phi_rn[i] = UniTensor()
        if len(functors) == 2 and mode == 0:
            for i in xrange(site_with_label_bond):
                self.rn_left_custom(clsfy_mps, i, functors[0], **kwargs)
            for i in xrange(self.len-1, site_with_label_bond, -1):
                self.rn_right_custom(clsfy_mps, i, functors[1], **kwargs)
        else:
            raise NotImplementedError

    def rn_left_custom(self, clsfy_mps, idx, functor, **kwargs):
        """"""
        self.phi_rn[idx] = functor(self, clsfy_mps, idx, **kwargs)
        self.rnb = max(self.rnb, idx+1)

    def rn_right_custom(self, clsfy_mps, idx, functor, **kwargs):
        """"""
        self.phi_rn[idx] = functor(self, clsfy_mps, idx, **kwargs)
        self.rnb = idx-1 if self.rnb == None else min(self.rnb, idx-1)


class ImageDataSet(object):
    """
    """
    def __init__(self, image_ndas, true_labels, dim_l=10, **kwargs):
        """"""
        self.size = len(image_ndas)
        self.data = []
        self.phi = []
        self.phi_rn = []
        self.label = []
        self.tl = []
        self.dl = dim_l
        for s in xrange(self.size):
            img_data = ImageData(ImageMPS(image_ndas[s], **kwargs), true_labels[s], dim_l)
            self.data.append(img_data)
            self.phi.append(img_data.phi)
            self.phi_rn.append(img_data.phi_rn)
            self.label.append(img_data.label)
            self.tl.append(img_data.tl)
        self.px = self.data[0].px if self.size > 0 else 0
        self.wd = self.data[0].wd if self.size > 0 else 0
        self.ht = self.data[0].ht if self.size > 0 else 0
        self.len = self.data[0].len if self.size > 0 else 0
        self.phys_lab_base = self.data[0].phys_lab_base if self.size > 0 else 1000

    def __getitem__(self, sample):
        return self.data[sample]
    
    def refresh_phi_rn(self, clsfy_mps):
        """"""
        for s in xrange(self.size):
            for i in xrange(self.len):
                self.phi_rn[s][i] = UniTensor()

        site_with_label_bond = clsfy_mps.sl
        for i in xrange(site_with_label_bond):
            self.update_phi_rn(clsfy_mps, i, True)
        for i in xrange(self.len-1, site_with_label_bond, -1):
            self.update_phi_rn(clsfy_mps, i, False)

    def renorm_phi_left(self, clsfy_mps, idx):
        """"""
        for s in xrange(self.size):
            if idx == 0:
                self.phi_rn[s][idx] = clsfy_mps[idx] * 1.;
            else:
                self.phi_rn[s][idx] = contract(self.phi_rn[s][idx-1], clsfy_mps[idx], False)
            self.phi_rn[s][idx] = contract(self.phi_rn[s][idx], self.phi[s][idx], False)
            if self.phi_rn[s][idx].inBondNum() > 1: self.phi_rn[s][idx].permute(1)
            self.data[s].rnb = max(self.data[s].rnb, idx+1)
        
    def renorm_phi_right(self, clsfy_mps, idx):
        """"""
        for s in xrange(self.size):
            if idx == self.len-1:
                self.phi_rn[s][idx] = clsfy_mps[idx] * 1.;
            else:
                self.phi_rn[s][idx] = contract(clsfy_mps[idx], self.phi_rn[s][idx+1], False)
            self.phi_rn[s][idx] = contract(self.phi_rn[s][idx], self.phi[s][idx], False)
            if self.phi_rn[s][idx].inBondNum() > 1: self.phi_rn[s][idx].permute(1)
            self.data[s].rnb = idx-1 if self.data[s].rnb == None else min(self.data[s].rnb, idx-1)
    
    def update_phi_rn(self, clsfy_mps, idx, left_rn):
        """"""
        if left_rn:
            self.renorm_phi_left(clsfy_mps, idx)
        else:
            self.renorm_phi_right(clsfy_mps, idx)

    def refresh_rn_custom(self, clsfy_mps, functors=[], mode=0, **kwargs):
        """"""
        for s in xrange(self.size):
            for i in xrange(self.len):
                self.phi_rn[s][i] = UniTensor()

        site_with_label_bond = clsfy_mps.sl
        if len(functors) == 2 and mode == 0:
            for i in xrange(site_with_label_bond):
                self.update_rn_custom(clsfy_mps, i, functors, mode=mode, flag="left", **kwargs)
            for i in xrange(self.len-1, site_with_label_bond, -1):
                self.update_rn_custom(clsfy_mps, i, functors, mode=mode, flag="right", **kwargs)
        else:
            raise NotImplementedError

    def rn_left_custom(self, clsfy_mps, idx, functor, **kwargs):
        """"""
        for s in xrange(self.size):
            self.phi_rn[s][idx] = functor(self, clsfy_mps, idx, **kwargs)
            self.data[s].rnb = max(self.data[s].rnb, idx+1)

    def rn_right_custom(self, clsfy_mps, idx, functor, **kwargs):
        """"""
        for s in xrange(self.size):
            self.phi_rn[s][idx] = functor(self, clsfy_mps, idx, **kwargs)
            self.data[s].rnb = idx-1 if self.data[s].rnb == None else min(self.data[s].rnb, idx-1)

    def update_rn_custom(self, clsfy_mps, idx, functors=[], mode=0, flag=None, **kwargs):
        """"""
        if len(functors) == 2 and mode == 0:
            if flag[0] == "l":
                self.rn_left_custom(clsfy_mps, idx, functors[0], **kwargs)
            elif flag[0] == "r":
                self.rn_right_custom(clsfy_mps, idx, functors[1], **kwargs)
            else:
                raise AttributeError, "Unavailable flag in update_rn_custom()"
        else:
            raise NotImplementedError
