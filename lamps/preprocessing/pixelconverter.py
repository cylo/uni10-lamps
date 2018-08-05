import numpy as np

def grayscale_to_spinor(x):
    ## norm = 1
    return [np.cos(np.pi/2. * x), np.sin(np.pi/2. * x)]

def spinor_to_grayscale(phi):
    return np.arccos(phi[0]) * 2./np.pi

def grayscale_to_linear(x, slope=0.25, black_ground=False):
    ## norm != 1
    if black_ground:
        return [1., (1.-x) * slope]
    return [1., x * slope]

def linear_to_grayscale(phi, slope=0.25, black_ground=False):
    if black_ground:
        return (phi[1]*(1./slope) - 1.) * (-1.)
    return phi[1]*(1./slope)/phi[0]
