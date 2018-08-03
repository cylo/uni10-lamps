import numpy as np

def grayscale_to_spinor(x):
    ## norm = 1
    return [np.cos(np.pi/2. * x/255.), np.sin(np.pi/2. * x/255.)]

def spinor_to_grayscale(phi):
    return np.arccos(phi[0]) * 2./np.pi * 255.

def grayscale_to_linear(x, slope=0.25, black_ground=False):
    ## norm != 1
    if black_ground:
        return [1., (1.-(x/255.)) * slope]
    return [1., x/255. * slope]

def linear_to_grayscale(phi, slope=0.25, black_ground=False):
    if black_ground:
        return (phi[1]*(1./slope) - 1.) * (-255.)
    return phi[1]*(1./slope)*255./phi[0]
