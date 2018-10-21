import os
from imagemps import *
from utils import tensor_to_ndarray
import shared

class ClassifierMPS(MPS):
    """
    """
    def __init__(self, width, height, networks, net_dir=shared.NETDIR+"/clsfy-mps",
                 chi_max=10, dim_label=10, site_with_label_bond=1,
                 dim_virt=1, dim_phys=2, bdry_dummy=True, phys_lab_base=1000):
        """"""
        self.wd = width
        self.ht = height
        self.px = width*height
        MPS.__init__(self, self.px)
        
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

        self.net = networks
        self.net_dir = net_dir
        self.load_networks(net_dir)

    def load_networks(self, net_dir=None):
        """"""
        if net_dir:
            self.net_dir = net_dir
        self.net["decision_fn_left"] = Network(self.net_dir + "/decision_fn_left.net")
        self.net["decision_fn_right"] = Network(self.net_dir + "/decision_fn_right.net")
        self.net["measurement"] = Network(self.net_dir + "/measurement.net")
        self.net["lvec_update"] = Network(self.net_dir + "/lvec_update.net")

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

    def dcsfunc_custom(self, img_data, functor_df, refresh_phi_rn=True, functors_rn=[], **kwargs):
        """"""
        assert self.__bdry_dummy, 'Boundary dummies not found.'
        if site < 0: site = self.sl
        if refresh_phi_rn: img_data.refresh_rn_custom(self, functors_rn, **kwargs)
        df = functor_df(self, img_data, site)
        return df

    def error_function(self, img_data, dcsfn=None, **kwargs):
        """"""
        if dcsfn:
            assert self.__bdry_dummy, 'Boundary dummies not found.'
            ef = img_data.tl + (-1.)*dcsfn(self, img_data, **kwargs)
        else:
            ef = img_data.tl + (-1.)*self.decision_function(img_data, **kwargs)
        return ef

    def predict(self, img_data, verbose=False, dcsfn=None, refresh_phi_rn=True, **kwargs):
        """"""
        if dcsfn:
            if refresh_phi_rn: img_data.refresh_phi_rn(self)
            dfb = dcsfn(self, img_data, **kwargs).getBlock()
        else:
            dfb = self.decision_function(img_data, refresh_phi_rn=refresh_phi_rn, **kwargs).getBlock()
        df = tensor_to_ndarray(dfb) if shared.OLD_VER else exportElem(dfb)
        if verbose:
            return df
        return np.argmax(df)
    
    def accuracy(self, image_set, **kwargs):
        """"""
        correct = []
        for s in xrange(image_set.size):
            correct.append(self.predict(image_set[s], **kwargs) == image_set[s].label)
        return sum(correct)/float(len(correct))
    
    def overlap(self, other):
        """"""
        same_struct = (self.len == other.len)
        same_struct *= (self.dp == other.dp)
        same_struct *= (self.sl == other.sl)
        same_struct *= (self.w[self.sl].label()[-1] == other.w[other.sl].label()[-1])
        same_struct *= (self._MPS__label_ordered and other._MPS__label_ordered)
        if same_struct:
            expv = UniTensor(); norm = UniTensor()
            for i in xrange(self.len-1):
                wsi = self.w[i] * 1.
                woi = other.w[i] * 1.
                _ , labs = in_out_bonds(wsi)
                if i < self.len-1: labs[1][0] *= -1
                if i > 0: labs[0][0] *= -1
                woi.setLabel(labs[0]+labs[1])
                expv = wsi * woi if i == 0 else wsi * expv * woi

                wsd = self.w[i] * 1.
                wsd.setLabel(labs[0]+labs[1])
                norm = wsi * wsd if i == 0 else wsi * norm * wsd
            return expv[0]/norm[0]
        else:
            print "MPS to be calculated have different structures."
            raise RuntimeError

    def feature_map(self, label, func=spinor_to_grayscale, snake=False):
        """"""
        assert self.__bdry_dummy, 'Boundary dummies not found.'
        
        # contract the selected label
        lab_vec = true_label_vec(label, self.dl)
        lab_vec.setLabel([-10])
        assert -10 in self.w[self.sl].label(), 'Site with label bond index error.'
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

    def reconstruct_sample(self, img_data, copy_px=1,
                           to_mps=grayscale_to_spinor, to_img=spinor_to_grayscale, snake=False):
        """"""
        assert self.__bdry_dummy and img_data._ImageMPS__bdry_dummy, 'Boundary dummies not found.'

        # contract the selected label
        lab_vec = true_label_vec(img_data.label, self.dl)
        lab_vec.setLabel([-10])
        swl = contract(lab_vec, self.w[self.sl])
        swl.permute(self.w[self.sl].inBondNum()-1)

        # define operators for measurement
        (phi_w, phi_wp, phi_p, phi_pg, phi_g, phi_gd, phi_d, phi_db, phi_b), \
            (NW, NWP, NP, NPG, NG, NGD, ND, NDB, NB) = self.grayscale_colors(to_mps)

        # construct left-vec/right-vec buffer
        lv = [UniTensor() for _ in xrange(self.len)]
        rv = [UniTensor() for _ in xrange(self.len)]
        for i in xrange(copy_px+1):
            wi = swl * 1. if i == self.sl else self.w[i] * 1.
            wd = wi * 1.  # w^dagger
            _ , labs = in_out_bonds(wi)
            labs[1][0] *= -1
            if i > 0:
                labs[0][0] *= -1
                labs[1][1] *= -1
            wd.setLabel(labs[0]+labs[1])
            if i == 0:
                lv[i] = wi * wd
            else:
                NI = otimes(img_data[i], img_data[i]); NI.permute(1)
                NI.setLabel([wi.label()[2], wd.label()[2]])
                lv[i] = wi * lv[i-1] * NI * wd

        for i in xrange(self.len-1, copy_px, -1):
            wi = swl * 1. if i == self.sl else self.w[i] * 1.
            wd = wi * 1.
            _ , labs = in_out_bonds(wi)
            labs[0][0] *= -1
            if i < self.len-1: labs[1][0] *= -1
            wd.setLabel(labs[0]+labs[1])
            rv[i] = wi * wd if i == self.len-1 else wi * rv[i+1] * wd

        # generative sampling
        res = [img_data[i] * 1. for i in xrange(1, copy_px+1)]
        netm = self.net["measurement"]
        netl = self.net["lvec_update"]
        for i in xrange(copy_px+1, self.px+1):
            wi = swl * 1. if i == self.sl else self.w[i] * 1.
            wd = wi * 1.
            netm.putTensor("WI", wi)
            netm.putTensor("WD", wd)
            netm.putTensor("LV", lv[i-1])
            netm.putTensor("RV", rv[i+1])
            netm.putTensor("OP", NW)
            nw = netm.launch()[0]
            netm.putTensor("OP", NWP)
            nwp = netm.launch()[0]
            netm.putTensor("OP", NP)
            np_ = netm.launch()[0]
            netm.putTensor("OP", NPG)
            npg = netm.launch()[0]
            netm.putTensor("OP", NG)
            ng = netm.launch()[0]
            netm.putTensor("OP", NGD)
            ngd = netm.launch()[0]
            netm.putTensor("OP", ND)
            nd = netm.launch()[0]
            netm.putTensor("OP", NDB)
            ndb = netm.launch()[0]
            netm.putTensor("OP", NB)
            nb = netm.launch()[0]

            netl.putTensor("WI", wi)
            netl.putTensor("WD", wd)
            netl.putTensor("LV", lv[i-1])
            max_color = max(nw, nwp, np_, npg, ng, ngd, nd, ndb, nb)
            if max_color == nw:
                res += [phi_w * 1.]
                netl.putTensor("OP", NW)
            elif max_color == nwp:
                res += [phi_wp * 1.]
                netl.putTensor("OP", NWP)
            elif max_color == np_:
                res += [phi_p * 1.]
                netl.putTensor("OP", NP)
            elif max_color == npg:
                res += [phi_pg * 1.]
                netl.putTensor("OP", NPG)
            elif max_color == ng:
                res += [phi_g * 1.]
                netl.putTensor("OP", NG)
            elif max_color == ngd:
                res += [phi_gd * 1.]
                netl.putTensor("OP", NGD)
            elif max_color == nd:
                res += [phi_d * 1.]
                netl.putTensor("OP", ND)
            elif max_color == ndb:
                res += [phi_db * 1.]
                netl.putTensor("OP", NDB)
            else:
                res += [phi_b * 1.]
                netl.putTensor("OP", NB)
            lv[i] = netl.launch()

        # convert mps into image
        img_mps = ImageMPS(np.zeros((self.wd, self.ht)))
        img_mps.import_tensors(res)
        if snake:
            return img_mps.mps_to_img_snake(to_img)
        return img_mps.mps_to_img(to_img)

    def generate_sample(self, image_set, label=-1, seed_ratio=0.3, seed_from_label=True, num_chimera=3,
                        to_mps=grayscale_to_spinor, to_img=spinor_to_grayscale, snake=False):
        """"""
        assert self.__bdry_dummy and image_set[0]._ImageMPS__bdry_dummy, 'Boundary dummies not found.'

        # contract the selected label
        if label < 0:
            label = np.random.choice(range(self.dl))
        lab_vec = true_label_vec(label, self.dl)
        lab_vec.setLabel([-10])
        swl = contract(lab_vec, self.w[self.sl])
        swl.permute(self.w[self.sl].inBondNum()-1)

        # define operators for measurement
        (phi_w, phi_wp, phi_p, phi_pg, phi_g, phi_gd, phi_d, phi_db, phi_b), \
            (NW, NWP, NP, NPG, NG, NGD, ND, NDB, NB) = self.grayscale_colors(to_mps)

        # seed crystals
        seed_idx = np.sort(np.random.choice(range(1, self.px+1), int(self.px*seed_ratio), replace=False))
        seed_phi = {}
        seed_NI = {}

        sample_idx = []
        # random pick num_chimera samples
        for n in xrange(num_chimera):
            to_pick = False
            while not to_pick:
                s = np.random.choice(range(image_set.size))
                if seed_from_label:
                    to_pick = (image_set[s].label == label)
                else:
                    to_pick = True
            sample_idx.append(s)

        for i in seed_idx:
            s = np.random.choice(sample_idx)
            seed_phi[i] = image_set[s][i] * 1.
            NI = otimes(image_set[s][i], image_set[s][i]); NI.permute(1)
            seed_NI[i] = NI * 1.

        # construct left-vec/right-vec buffer
        lv = [UniTensor() for _ in xrange(self.len)]
        rv = [UniTensor() for _ in xrange(self.len)]
        for i in xrange(1):
            wi = self.w[i] * 1.
            wd = wi * 1.  # w^dagger
            _ , labs = in_out_bonds(wi)
            labs[1][0] *= -1
            wd.setLabel(labs[0]+labs[1])
            lv[i] = wi * wd

        for i in xrange(self.len-1, 0, -1):
            wi = swl * 1. if i == self.sl else self.w[i] * 1.
            wd = wi * 1.
            _ , labs = in_out_bonds(wi)
            labs[0][0] *= -1
            if i < self.len-1: labs[1][0] *= -1
            wd.setLabel(labs[0]+labs[1])
            if i in seed_idx:
                lab_wd = list(wd.label()[:-1]) + [wd.label()[-1] * (-1)]
                wd.setLabel(lab_wd)
                NI = seed_NI[i]
                # print wi.label(), wd.label()
                NI.setLabel([wi.label()[2], wd.label()[2]])
                rv[i] = wi * rv[i+1] * NI * wd
            elif i == self.len-1:
                rv[i] = wi * wd
            else:
                rv[i] =  wi * rv[i+1] * wd

        # generative sampling
        res = []
        netm = self.net["measurement"]
        netl = self.net["lvec_update"]
        for i in xrange(1, self.px+1):
            wi = swl * 1. if i == self.sl else self.w[i] * 1.
            wd = wi * 1.
            # if i in seed_idx:
            #     netl.putTensor("WI", wi)
            #     netl.putTensor("WD", wd)
            #     netl.putTensor("LV", lv[i-1])
            #     res += [seed_phi[i] * 1.]
            #     netl.putTensor("OP", seed_NI[i])
            # else:
            netm.putTensor("WI", wi)
            netm.putTensor("WD", wd)
            netm.putTensor("LV", lv[i-1])
            netm.putTensor("RV", rv[i+1])
            netm.putTensor("OP", NW)
            nw = netm.launch()[0]
            netm.putTensor("OP", NWP)
            nwp = netm.launch()[0]
            netm.putTensor("OP", NP)
            np_ = netm.launch()[0]
            netm.putTensor("OP", NPG)
            npg = netm.launch()[0]
            netm.putTensor("OP", NG)
            ng = netm.launch()[0]
            netm.putTensor("OP", NGD)
            ngd = netm.launch()[0]
            netm.putTensor("OP", ND)
            nd = netm.launch()[0]
            netm.putTensor("OP", NDB)
            ndb = netm.launch()[0]
            netm.putTensor("OP", NB)
            nb = netm.launch()[0]

            netl.putTensor("WI", wi)
            netl.putTensor("WD", wd)
            netl.putTensor("LV", lv[i-1])
            max_color = max(nw, nwp, np_, npg, ng, ngd, nd, ndb, nb)
            if max_color == nw:
                res += [phi_w * 1.]
                netl.putTensor("OP", NW)
            elif max_color == nwp:
                res += [phi_wp * 1.]
                netl.putTensor("OP", NWP)
            elif max_color == np_:
                res += [phi_p * 1.]
                netl.putTensor("OP", NP)
            elif max_color == npg:
                res += [phi_pg * 1.]
                netl.putTensor("OP", NPG)
            elif max_color == ng:
                res += [phi_g * 1.]
                netl.putTensor("OP", NG)
            elif max_color == ngd:
                res += [phi_gd * 1.]
                netl.putTensor("OP", NGD)
            elif max_color == nd:
                res += [phi_d * 1.]
                netl.putTensor("OP", ND)
            elif max_color == ndb:
                res += [phi_db * 1.]
                netl.putTensor("OP", NDB)
            else:
                res += [phi_b * 1.]
                netl.putTensor("OP", NB)
            lv[i] = netl.launch()

        # convert mps into image
        img_mps = ImageMPS(np.zeros((self.wd, self.ht)))
        img_mps.import_tensors(res)
        if snake:
            return img_mps.mps_to_img_snake(to_img)
        return img_mps.mps_to_img(to_img)

    def grayscale_colors(self, to_mps):
        """"""
        phi_w  = UniTensor([Bond(BD_IN, 2)]); phi_w.setElem(to_mps(0.00))  # white
        phi_wp = UniTensor([Bond(BD_IN, 2)]); phi_wp.setElem(to_mps(0.125))
        phi_p  = UniTensor([Bond(BD_IN, 2)]); phi_p.setElem(to_mps(0.25))  # pale
        phi_pg = UniTensor([Bond(BD_IN, 2)]); phi_pg.setElem(to_mps(0.375))
        phi_g  = UniTensor([Bond(BD_IN, 2)]); phi_g.setElem(to_mps(0.50))  # grey
        phi_gd = UniTensor([Bond(BD_IN, 2)]); phi_gd.setElem(to_mps(0.625))
        phi_d  = UniTensor([Bond(BD_IN, 2)]); phi_d.setElem(to_mps(0.75))  # dark
        phi_db = UniTensor([Bond(BD_IN, 2)]); phi_db.setElem(to_mps(0.875))
        phi_b  = UniTensor([Bond(BD_IN, 2)]); phi_b.setElem(to_mps(1.00))  # black
        NW  = otimes(phi_w, phi_w); NW.permute(1)
        NWP = otimes(phi_wp, phi_wp); NWP.permute(1)
        NP  = otimes(phi_p, phi_p); NP.permute(1)
        NPG = otimes(phi_pg, phi_pg); NPG.permute(1)
        NG  = otimes(phi_g, phi_g); NG.permute(1)
        NGD = otimes(phi_gd, phi_gd); NGD.permute(1)
        ND  = otimes(phi_d, phi_d); ND.permute(1)
        NDB = otimes(phi_db, phi_db); NDB.permute(1)
        NB  = otimes(phi_b, phi_b); NB.permute(1)
        return (phi_w, phi_wp, phi_p, phi_pg, phi_g, phi_gd, phi_d, phi_db, phi_b), \
            (NW, NWP, NP, NPG, NG, NGD, ND, NDB, NB)
