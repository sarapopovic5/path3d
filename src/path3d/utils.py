# colour deconvolution: if only used for upstream IHC, move to IHC specific module

import sys

import numpy as np
import skimage
from skimage.color import hed_from_rgb # stain seperation matrix. H+E+DAB



def quiet_progress(iterable, desc=None, total=None):
    """
    twdm stand-in: prints "done" instead of thousands of the progress bars
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    n = 0
    for item in iterable:
        n += 1
        yield item

    prefix = f"{desc}: " if desc is not None else ""
    if total is not None:
        print(f"{prefix}{n}/{total}", file=sys.stderr, flush=True)
    else:
        print(f"{prefix}{n}", file=sys.stderr, flush=True)


def deconvolve(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Needs to output H channel and E channel
    hed from rgb (convolution matrix) will give us DAB along with H and E channels. 
    Needed for IHC progression (channel to highlight protein stain)

    Returns triplet of the 3 separate channels
    
    To access: 
    h, e, dab = deconvolve(image_rgb), or

    result = deconcolve(image_rgb)
    h = result[0]
    """
    # use `skimage.color.separate_stains`

    # print(f"Input image: {image_rgb}")

    colourspace_img = skimage.color.separate_stains(image_rgb, hed_from_rgb, channel_axis=-1) 
   
    # print(f"Colourspace image: {colourspace_img}")

    H_channel = colourspace_img[:,:,0]
    E_channel = colourspace_img[:,:,1]
    DAB_channel = colourspace_img[:,:,2]

    # print(f"H channel: {H_channel}")
    # print(f"DAB channel: {DAB_channel}")

    # print(f"Deconvolution matrix: {hed_from_rgb}")

    return (H_channel, E_channel, DAB_channel)
    



