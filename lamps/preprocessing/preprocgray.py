import numpy as np
import skimage.transform as skt
import skimage.exposure as ske

def shrink_grayscale(image_nda, factor=2, **kwargs):
    """
    Shrink grayscale image
    """
    wd = image_nda.shape[0]//factor
    ht = image_nda.shape[1]//factor
    return skt.resize(image_nda, (wd, ht), preserve_range=True, **kwargs)

def enhance_grayscale(image_nda, cutoff=0.5, gain=10., **kwargs):
    """
    Enhance grayscale image
    """
    return ske.adjust_sigmoid(image_nda, cutoff=cutoff*255., gain=gain/255., **kwargs) * 255.

def preprocess_grayscale(image_ndas, shrink_factor=None, contrast_cutoff=None, contrast_gain=None,
                        functors=[]):
    """
    """
    if contrast_cutoff or contrast_gain:
        if contrast_cutoff and contrast_gain:
            func = lambda x: enhance_grayscale(x, contrast_cutoff, contrast_gain)
        elif contrast_cutoff:
            func = lambda x: enhance_grayscale(x, cutoff=contrast_cutoff)
        else:
            func = lambda x: enhance_grayscale(x, gain=contrast_gain)
        functors = [func] + functors
    
    if shrink_factor:
        functors = [lambda x: shrink_grayscale(x, shrink_factor)] + functors
    
    image_processed = []
    for s in xrange(len(image_ndas)):
        img = image_ndas[s]
        for f in functors:
            img = f(img)
        image_processed.append(img)

    return np.array(image_processed)
